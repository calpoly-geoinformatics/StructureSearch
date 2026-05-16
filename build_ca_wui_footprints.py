"""
build_ca_wui_footprints.py
==========================
Batch pre-fire building footprint compiler for CA WUI fire hindcast analysis.

For each fire in FIRE_INVENTORY:
  1. Fetch CAL FIRE burn perimeter (FRAP API)
  2. Query ohsome API for OSM buildings as of pre-fire date
  3. Download Overture Maps ML buildings (non-OSM sources) via S3
  4. Merge with IoU deduplication (OSM priority)
  5. Clip to burn perimeter
  6. If DINS parquet available → build DINS-centric unified dataset
     (real polygons where matched, circular estimates where not)
  7. Compute wall-to-wall separation + orientation metrics
  8. Save GeoPackage + summary JSON

Outputs per fire:
  <OUTPUT_ROOT>/<fire_slug>/
    <fire_slug>_prefire_buildings.gpkg      OSM + ML merged
    <fire_slug>_unified_structures.gpkg     DINS hybrid (if DINS available)
    <fire_slug>_summary.json               Run metadata + coverage stats
    <fire_slug>_source_map.png
    <fire_slug>_spacing.png

Usage:
  python build_ca_wui_footprints.py                  # all fires
  python build_ca_wui_footprints.py --fires Woolsey Kincade
  python build_ca_wui_footprints.py --resume          # skip already-completed
"""

import argparse
import json
import logging
import math
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely import wkb
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths and fire list here
# ─────────────────────────────────────────────────────────────────────

# Root directory for all fire outputs
OUTPUT_ROOT = Path(__file__).parent / "fires"

# Directory containing DINS parquet files.
# Expected filename convention: DINS_<fire_slug>.parquet
# e.g. DINS_Woolsey.parquet, DINS_LNU_Lightning_Complex.parquet
DINS_DIR = Path(__file__).parent / "DINS"

# Processing knobs
IOU_THRESHOLD  = 0.30   # IoU above which ML building is dropped as OSM duplicate
MATCH_M        = 25     # metres — DINS/footprint centroid match threshold
WALL_SEARCH_R  = 60     # metres — wall-to-wall gap search radius
MIN_AREA_M2    = 15     # m² — drop smaller polygons (noise/sheds)
AREA_FILTER    = 15     # minimum footprint area to include in spacing metrics

# DINS structure classes to exclude (minor structures, infra, agriculture)
DINS_EXCLUDE = {"Other Minor Structure", "Infrastructure", "Agriculture"}

# ── CAL FIRE FRAP endpoint ────────────────────────────────────────────
CALFIRE_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Fire_Perimeters_all/FeatureServer/0/query"
)
OHSOME_URL  = "https://api.ohsome.org/v1/elements/geometry"

# ─────────────────────────────────────────────────────────────────────
# FIRE INVENTORY
# Each entry: name, ignition_date (YYYY-MM-DD), calfire_where clause,
#             dins_slug (filename stem after "DINS_", or None)
# ─────────────────────────────────────────────────────────────────────
FIRE_INVENTORY = [
    dict(name="Woolsey",               date="2018-11-08",
         where="FIRE_NAME='WOOLSEY' AND YEAR_=2018",               dins="Woolsey"),
    dict(name="Kincade",               date="2019-10-23",
         where="FIRE_NAME='KINCADE' AND YEAR_=2019",               dins="Kincade"),
    dict(name="LNU_Lightning_Complex", date="2020-08-17",
         where="FIRE_NAME='LNU LIGHTNING COMPLEX' AND YEAR_=2020", dins="LNU_Lightning_Complex"),
    dict(name="CZU_Lightning_Complex", date="2020-08-16",
         where="FIRE_NAME='CZU LIGHTNING COMPLEX' AND YEAR_=2020", dins="CZU_Lightning_Complex"),
    dict(name="North_Complex",         date="2020-08-17",
         where="FIRE_NAME='NORTH COMPLEX' AND YEAR_=2020",         dins="North_Complex"),
    dict(name="Creek",                 date="2020-09-04",
         where="FIRE_NAME='CREEK' AND YEAR_=2020",                 dins="Creek"),
    dict(name="Glass",                 date="2020-09-27",
         where="FIRE_NAME='GLASS' AND YEAR_=2020",                 dins="Glass"),
    dict(name="Zogg",                  date="2020-09-27",
         where="FIRE_NAME='ZOGG' AND YEAR_=2020",                  dins="Zogg"),
    dict(name="Dixie",                 date="2021-07-13",
         where="FIRE_NAME='DIXIE' AND YEAR_=2021",                 dins="Dixie"),
    dict(name="Caldor",                date="2021-08-14",
         where="FIRE_NAME='CALDOR' AND YEAR_=2021",                dins="Caldor"),
    dict(name="McKinney",              date="2022-07-26",
         where="FIRE_NAME='MCKINNEY' AND YEAR_=2022",              dins="McKinney"),
    dict(name="Oak",                   date="2022-07-22",
         where="FIRE_NAME='OAK' AND YEAR_=2022",                   dins="Oak"),
    dict(name="Park",                  date="2024-07-24",
         where="FIRE_NAME='PARK' AND YEAR_=2024",                  dins="Park"),
    dict(name="Airport",               date="2024-09-09",
         where="FIRE_NAME='AIRPORT' AND YEAR_=2024",               dins="Airport"),
    dict(name="Palisades",             date="2025-01-07",
         where="FIRE_NAME='PALISADES' AND YEAR_=2025",             dins="Palisades"),
    dict(name="Eaton",                 date="2025-01-07",
         where="FIRE_NAME='EATON' AND YEAR_=2025",                 dins="Eaton"),
]

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# STEP 1 — Fetch fire perimeter
# ═════════════════════════════════════════════════════════════════════

def fetch_fire_perimeter(where_clause: str) -> shape:
    """
    Fetch the CAL FIRE burn perimeter polygon from the FRAP FeatureServer.
    Returns the largest polygon geometry (final perimeter).
    Raises RuntimeError if the fetch fails or returns no features.
    """
    params = {
        "where":     where_clause,
        "outFields": "FIRE_NAME,YEAR_,GIS_ACRES,ALARM_DATE",
        "f":         "geojson",
        "outSR":     "4326",
    }
    r = requests.get(CALFIRE_URL, params=params, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        raise RuntimeError(f"No CAL FIRE features for: {where_clause}")
    geoms = [shape(f["geometry"]) for f in feats if f.get("geometry")]
    return max(geoms, key=lambda g: g.area)


# ═════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch OSM buildings via ohsome API
# ═════════════════════════════════════════════════════════════════════

def fetch_osm_buildings(bbox: tuple, pre_fire_date: str) -> gpd.GeoDataFrame:
    """
    Query ohsome API for all OSM buildings as of pre_fire_date (YYYY-MM-DD).
    bbox: (xmin, ymin, xmax, ymax) WGS84
    Returns GeoDataFrame with source='openstreetmap' and source_date timestamps.
    """
    xmin, ymin, xmax, ymax = bbox
    params = {
        "bboxes":     f"{xmin},{ymin},{xmax},{ymax}",
        "filter":     "building=* and geometry:polygon",
        "time":       pre_fire_date,
        "properties": "tags,metadata",
    }
    log.info("  ohsome query: bbox=%s  date=%s", params["bboxes"], pre_fire_date)
    resp = requests.post(OHSOME_URL, data=params, timeout=300)
    if resp.status_code != 200:
        log.warning("  ohsome returned %d — %s", resp.status_code, resp.text[:200])
        return gpd.GeoDataFrame(columns=["geometry", "source", "source_date"], crs="EPSG:4326")

    fc = resp.json()
    n  = len(fc.get("features", []))
    log.info("  ohsome: %d features returned", n)
    if n == 0:
        return gpd.GeoDataFrame(columns=["geometry", "source", "source_date"], crs="EPSG:4326")

    # Auto-detect timestamp field
    sample = fc["features"][0].get("properties", {})
    ts_key = next(
        (k for k in ["@snapshotTimestamp", "@timestamp", "@validFrom", "timestamp"]
         if k in sample),
        None,
    )
    log.info("  ohsome timestamp field: %s", ts_key)

    records = []
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            continue

        osm_id = props.get("@osmId", "")
        records.append({
            "geometry":    geom,
            "source":      "openstreetmap",
            "source_date": props.get(ts_key) if ts_key else None,
            "osm_id":      osm_id.split("/")[-1] if "/" in osm_id else osm_id,
            "building":    props.get("building", "yes"),
            "name":        props.get("name"),
            "height":      props.get("height"),
            "levels":      props.get("building:levels"),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["source_date"] = pd.to_datetime(gdf["source_date"], utc=True, errors="coerce")
    return gdf


# ═════════════════════════════════════════════════════════════════════
# STEP 3 — Fetch Overture ML buildings (non-OSM) via S3
# ═════════════════════════════════════════════════════════════════════

def fetch_overture_buildings(bbox: tuple) -> gpd.GeoDataFrame:
    """
    Download non-OSM ML buildings from the latest Overture Maps release on S3.
    Returns GeoDataFrame with source = Overture dataset name (e.g. 'microsoft-buildings').
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    from urllib.request import urlopen

    xmin, ymin, xmax, ymax = bbox

    # Discover latest release
    with urlopen("https://stac.overturemaps.org/catalog.json") as r:
        latest = json.load(r)["latest"]
    log.info("  Overture release: %s", latest)

    s3_path = f"overturemaps-us-west-2/release/{latest}/theme=buildings/type=building/"
    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    dataset = ds.dataset(s3_path, filesystem=s3)

    bbox_filter = (
        (pc.field("bbox", "xmin") < xmax) & (pc.field("bbox", "xmax") > xmin) &
        (pc.field("bbox", "ymin") < ymax) & (pc.field("bbox", "ymax") > ymin)
    )
    table = dataset.to_table(filter=bbox_filter)
    log.info("  Overture: %d total rows retrieved", table.num_rows)

    records = []
    tally   = {}
    for row in table.to_pylist():
        sources = row.get("sources") or []
        if not sources:
            continue
        primary = sources[0].get("dataset", "").lower()
        if "openstreetmap" in primary:
            continue          # already captured via ohsome
        try:
            geom = wkb.loads(bytes(row["geometry"]))
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        records.append({
            "geometry":    geom,
            "source":      primary,
            "source_date": pd.NaT,
            "osm_id":      None,
            "building":    "yes",
            "name":        None,
            "height":      None,
            "levels":      None,
        })
        tally[primary] = tally.get(primary, 0) + 1

    for src, cnt in sorted(tally.items(), key=lambda x: -x[1]):
        log.info("    %-45s %6d", src, cnt)

    return gpd.GeoDataFrame(records, crs="EPSG:4326") if records else \
           gpd.GeoDataFrame(columns=["geometry","source","source_date"], crs="EPSG:4326")


# ═════════════════════════════════════════════════════════════════════
# STEP 4 — Merge: OSM priority, ML gap-fill
# ═════════════════════════════════════════════════════════════════════

def remove_ml_duplicates(
    gdf_osm: gpd.GeoDataFrame,
    gdf_ml:  gpd.GeoDataFrame,
    iou_threshold: float = 0.30,
) -> gpd.GeoDataFrame:
    """Drop ML buildings that overlap an OSM footprint above the IoU threshold."""
    if len(gdf_ml) == 0 or len(gdf_osm) == 0:
        return gdf_ml

    sindex = gdf_osm.sindex
    keep   = np.ones(len(gdf_ml), dtype=bool)

    for i, row in enumerate(gdf_ml.itertuples()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            keep[i] = False
            continue
        for j in sindex.intersection(geom.bounds):
            ogeom = gdf_osm.iloc[j].geometry
            if not geom.intersects(ogeom):
                continue
            inter = geom.intersection(ogeom).area
            union = geom.union(ogeom).area
            if union > 0 and (inter / union) >= iou_threshold:
                keep[i] = False
                break

    return gdf_ml[keep].reset_index(drop=True)


def merge_and_clip(
    gdf_osm:         gpd.GeoDataFrame,
    gdf_ml:          gpd.GeoDataFrame,
    fire_perimeter,
    iou_threshold:   float = 0.30,
) -> gpd.GeoDataFrame:
    """Deduplicate, concatenate, clip to perimeter, drop tiny polygons."""
    log.info("  Deduplicating ML buildings (IoU > %.2f)…", iou_threshold)
    gdf_ml_dd = remove_ml_duplicates(gdf_osm, gdf_ml, iou_threshold)
    log.info("  Dropped %d ML duplicates, kept %d", len(gdf_ml) - len(gdf_ml_dd), len(gdf_ml_dd))

    gdf_all = pd.concat([gdf_osm, gdf_ml_dd], ignore_index=True)
    gdf_all = gpd.GeoDataFrame(gdf_all, crs="EPSG:4326")
    gdf_all = gdf_all[gdf_all.geometry.notna() & ~gdf_all.geometry.is_empty
                      & gdf_all.geometry.is_valid].copy()

    fire_gdf = gpd.GeoDataFrame(geometry=[fire_perimeter], crs="EPSG:4326")
    gdf_all  = gpd.clip(gdf_all, fire_gdf).reset_index(drop=True)
    log.info("  After clip: %d buildings inside perimeter", len(gdf_all))
    return gdf_all


# ═════════════════════════════════════════════════════════════════════
# STEP 5 — Spatial metrics (wall-to-wall + orientation)
# ═════════════════════════════════════════════════════════════════════

def compute_orientation(geom) -> float:
    """Long-axis orientation of minimum rotated rectangle (0–180°)."""
    try:
        mbr    = geom.minimum_rotated_rectangle
        coords = list(mbr.exterior.coords)
        edges  = [(math.hypot(coords[i+1][0]-coords[i][0],
                              coords[i+1][1]-coords[i][1]),
                   coords[i+1][0]-coords[i][0],
                   coords[i+1][1]-coords[i][1])
                  for i in range(len(coords)-1)]
        edges.sort(reverse=True)
        _, dx, dy = edges[0]
        return math.degrees(math.atan2(dy, dx)) % 180
    except Exception:
        return np.nan


def compute_spatial_metrics(gdf_utm: gpd.GeoDataFrame,
                            search_r: float = WALL_SEARCH_R) -> gpd.GeoDataFrame:
    """
    Add to gdf_utm (projected, metres):
      area_m2         — footprint area
      orientation_deg — MRR long-axis angle
      min_wall_wall_m — smallest wall-to-wall gap to any neighbour within search_r
      nn_dist_m       — centroid-to-centroid nearest-neighbour distance
    """
    from tqdm.auto import tqdm

    gdf = gdf_utm.copy()
    gdf["area_m2"] = gdf.geometry.area

    log.info("  Computing orientations…")
    gdf["orientation_deg"] = [compute_orientation(g) for g in gdf.geometry]

    polys = gdf.geometry.values
    pts   = np.array([[g.centroid.x, g.centroid.y] for g in polys])
    tree  = STRtree(polys)
    kd    = cKDTree(pts)

    min_ww = np.full(len(gdf), np.inf)
    nn_cc  = np.full(len(gdf), np.inf)

    # Centroid-to-centroid (fast, via KDTree)
    dists, _ = kd.query(pts, k=2)
    nn_cc    = dists[:, 1]
    nn_cc[np.isinf(nn_cc)] = np.nan

    log.info("  Computing wall-to-wall gaps (search_r=%.0f m)…", search_r)
    for i in tqdm(range(len(gdf)), desc="wall-to-wall", leave=False):
        Pi    = polys[i]
        cands = tree.query(Pi.buffer(search_r))
        for j in cands:
            if j == i:
                continue
            d = Pi.distance(polys[j])
            if d < min_ww[i]:
                min_ww[i] = d
    min_ww[np.isinf(min_ww)] = np.nan

    gdf["min_wall_wall_m"] = min_ww
    gdf["nn_dist_m"]       = nn_cc
    return gdf


# ═════════════════════════════════════════════════════════════════════
# STEP 6 — DINS hybrid (if DINS parquet available)
# ═════════════════════════════════════════════════════════════════════

def load_dins(dins_path: Path) -> gpd.GeoDataFrame:
    """Load DINS parquet, rebuild WGS84 point geometry from LAT/LON columns."""
    gdf_raw = gpd.read_parquet(dins_path)
    mask    = ~gdf_raw["STRUCTUREC"].isin(DINS_EXCLUDE)
    df      = gdf_raw[mask].copy()
    df["geometry"] = [Point(r.LONGITUDE, r.LATITUDE) for r in df.itertuples()]
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def build_dins_hybrid(
    gdf_all:  gpd.GeoDataFrame,   # merged ML+OSM, WGS84
    gdf_dins: gpd.GeoDataFrame,   # DINS points, WGS84
    match_m:  float = MATCH_M,
) -> gpd.GeoDataFrame:
    """
    Unified structure dataset anchored to DINS locations.
    Returns GeoDataFrame in EPSG:32610 with geometry_source flag.
    """
    gdf_fp   = gdf_all.to_crs("EPSG:32610").copy()
    gdf_fp["area_m2"] = gdf_fp.geometry.area
    gdf_d    = gdf_dins.to_crs("EPSG:32610").copy()

    fp_cents = np.array([[g.centroid.x, g.centroid.y] for g in gdf_fp.geometry])
    d_pts    = np.array([[g.x, g.y] for g in gdf_d.geometry])
    tree     = cKDTree(fp_cents)
    nn_dists, nn_idx = tree.query(d_pts, k=1)

    # Per-class median areas from matched buildings
    struct_areas = {}
    for i, row in enumerate(gdf_d.itertuples()):
        if nn_dists[i] <= match_m:
            struct_areas.setdefault(row.STRUCTUREC, []).append(
                float(gdf_fp.iloc[nn_idx[i]]["area_m2"])
            )
    struct_medians = {cls: float(np.median(v)) for cls, v in struct_areas.items()}
    overall_median = float(np.median([a for v in struct_areas.values() for a in v])) \
                     if struct_areas else 100.0

    records   = []
    n_matched = 0
    n_est     = 0
    for i, row in enumerate(gdf_d.itertuples()):
        dist, fp_i = nn_dists[i], nn_idx[i]
        if dist <= match_m:
            fp      = gdf_fp.iloc[fp_i]
            geom    = fp.geometry
            area    = float(fp["area_m2"])
            gsrc    = "matched_footprint"
            fpsrc   = fp["source"]
            n_matched += 1
        else:
            med    = struct_medians.get(row.STRUCTUREC, overall_median)
            geom   = row.geometry.buffer(math.sqrt(med / math.pi))
            area   = med
            gsrc   = "estimated_centroid"
            fpsrc  = "estimated"
            n_est += 1
        records.append({
            "geometry":        geom,
            "STRUCTUREC":      row.STRUCTUREC,
            "DAMAGE":          row.DAMAGE,
            "LATITUDE":        row.LATITUDE,
            "LONGITUDE":       row.LONGITUDE,
            "area_m2":         area,
            "fp_source":       fpsrc,
            "geometry_source": gsrc,
            "nn_dist_m":       float(dist),
        })

    gdf_u = gpd.GeoDataFrame(records, crs="EPSG:32610")
    match_pct = n_matched / max(len(gdf_u), 1) * 100
    log.info("  DINS hybrid: %d matched (%.1f%%), %d estimated",
             n_matched, match_pct, n_est)
    return gdf_u, match_pct


# ═════════════════════════════════════════════════════════════════════
# STEP 7 — Plots
# ═════════════════════════════════════════════════════════════════════

def save_source_map(gdf_all, fire_perimeter, out_path):
    SOURCE_COLORS = {
        "openstreetmap":         "#2196F3",
        "microsoft-buildings":   "#FF6B35",
        "google-open-buildings": "#4CAF50",
        "esri-buildings":        "#9C27B0",
    }
    def get_color(src):
        src = (src or "").lower()
        for k, c in SOURCE_COLORS.items():
            if k in src: return c
        return "#9E9E9E"

    fig, ax = plt.subplots(figsize=(12, 8))
    gdf_all.plot(ax=ax, color=gdf_all["source"].apply(get_color),
                 edgecolor="none", alpha=0.7)
    gpd.GeoSeries([fire_perimeter]).plot(ax=ax, facecolor="none",
                                         edgecolor="red", linewidth=1, linestyle="--")
    patches = [mpatches.Patch(color=get_color(s), label=f"{s} ({n:,})")
               for s, n in gdf_all["source"].value_counts().items()]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                  linestyle="--", label="Burn perimeter"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title("Pre-Fire Building Footprints — Source Map", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_spacing_plot(gdf_utm, out_path, fire_name):
    ww = gdf_utm["min_wall_wall_m"].dropna()
    cc = gdf_utm["nn_dist_m"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(ww[ww < 60], bins=60, color="#E53935", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.median(ww), color="darkred", linestyle="--", linewidth=1.5,
                    label=f"Median: {np.median(ww):.1f} m")
    for x, lbl in [(3, "3 m"), (7.6, "7.6 m"), (15, "15 m")]:
        axes[0].axvline(x, color="orange", linestyle=":", linewidth=1, label=lbl)
    axes[0].set_xlabel("Min wall-to-wall gap (m)")
    axes[0].set_title("Wall-to-Wall Separation")
    axes[0].legend(fontsize=7)

    axes[1].hist(cc[cc < 120], bins=60, color="coral", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.median(cc), color="darkred", linestyle="--", linewidth=1.5,
                    label=f"Median: {np.median(cc):.1f} m")
    axes[1].set_xlabel("Centroid-to-centroid distance (m)")
    axes[1].set_title("Centroid Spacing (reference)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Building Separation — {fire_name.replace('_', ' ')} (pre-fire)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
# MAIN PER-FIRE PROCESSOR
# ═════════════════════════════════════════════════════════════════════

def process_fire(fire: dict, output_root: Path, dins_dir: Path,
                 resume: bool = False) -> dict:
    """
    Run the full footprint pipeline for one fire.
    Returns a summary dict (also written to <slug>_summary.json).
    """
    slug     = fire["name"].lower()
    out_dir  = output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{slug}_summary.json"
    if resume and summary_path.exists():
        log.info("  [%s] already completed — skipping (--resume)", fire["name"])
        return json.loads(summary_path.read_text())

    t0      = time.time()
    summary = {
        "fire":         fire["name"],
        "ignition_date": fire["date"],
        "processed_at": datetime.utcnow().isoformat(),
        "status":       "failed",
    }

    try:
        log.info("=" * 60)
        log.info("FIRE: %s  (%s)", fire["name"], fire["date"])
        log.info("=" * 60)

        # ── 1. Perimeter ─────────────────────────────────────────────
        log.info("[1/7] Fetching CAL FIRE perimeter…")
        perimeter = fetch_fire_perimeter(fire["where"])
        pb   = perimeter.bounds
        # bbox = perimeter bounds + 500m buffer (~0.005°)
        bbox = (pb[0]-0.005, pb[1]-0.005, pb[2]+0.005, pb[3]+0.005)
        area_km2 = perimeter.area * 111**2
        log.info("  Perimeter: %.0f km²  bbox: %s", area_km2, bbox)

        # ── 2. OSM via ohsome ─────────────────────────────────────────
        log.info("[2/7] Fetching OSM buildings (pre-fire)…")
        pre_date = (datetime.strptime(fire["date"], "%Y-%m-%d")
                    - timedelta(days=1)).strftime("%Y-%m-%d")
        gdf_osm  = fetch_osm_buildings(bbox, pre_date)
        log.info("  OSM: %d buildings", len(gdf_osm))

        # ── 3. Overture ML ────────────────────────────────────────────
        log.info("[3/7] Fetching Overture ML buildings…")
        gdf_ml = fetch_overture_buildings(bbox)
        log.info("  ML: %d non-OSM buildings", len(gdf_ml))

        # ── 4. Merge + clip ───────────────────────────────────────────
        log.info("[4/7] Merging and clipping to perimeter…")
        gdf_all = merge_and_clip(gdf_osm, gdf_ml, perimeter, IOU_THRESHOLD)
        gdf_all["source_date"] = pd.to_datetime(
            gdf_all.get("source_date"), utc=True, errors="coerce"
        )

        # ── 5. Spatial metrics ────────────────────────────────────────
        log.info("[5/7] Computing spatial metrics…")
        gdf_utm = gdf_all.to_crs("EPSG:32610").copy()
        gdf_utm = gdf_utm[gdf_utm.geometry.area >= MIN_AREA_M2].copy()
        gdf_utm = compute_spatial_metrics(gdf_utm, search_r=WALL_SEARCH_R)

        ww = gdf_utm["min_wall_wall_m"].dropna()
        log.info("  Wall-to-wall  median=%.1f m  p10=%.1f m  <3m: %.1f%%",
                 np.median(ww), np.percentile(ww, 10),
                 (ww < 3).mean() * 100)

        # ── 6. DINS hybrid (if available) ─────────────────────────────
        dins_path    = dins_dir / f"DINS_{fire['dins']}.parquet"
        dins_avail   = dins_path.exists()
        match_pct    = None
        unified_path = None

        if dins_avail:
            log.info("[6/7] Building DINS-centric unified dataset…")
            gdf_dins = load_dins(dins_path)
            log.info("  DINS: %d primary structures", len(gdf_dins))
            gdf_unified, match_pct = build_dins_hybrid(
                gdf_all.to_crs("EPSG:4326"), gdf_dins, match_m=MATCH_M
            )
            # Add wall-to-wall to unified dataset too
            gdf_unified = compute_spatial_metrics(gdf_unified, search_r=WALL_SEARCH_R)
            unified_path = out_dir / f"{slug}_unified_structures.gpkg"
            gdf_unified.to_crs("EPSG:4326").to_file(unified_path, driver="GPKG")
            log.info("  Unified saved → %s", unified_path)
        else:
            log.info("[6/7] No DINS file at %s — skipping hybrid step", dins_path)

        # ── 7. Save outputs ───────────────────────────────────────────
        log.info("[7/7] Saving outputs…")

        # Transfer metrics back to WGS84 GDF for the main save
        metrics = ["area_m2", "orientation_deg", "min_wall_wall_m", "nn_dist_m"]
        gdf_save = gdf_all.copy()
        for col in metrics:
            if col in gdf_utm.columns:
                gdf_save = gdf_save.merge(
                    gdf_utm[[col]].reset_index(),
                    left_index=True, right_on="index", how="left"
                ).drop(columns="index", errors="ignore")

        gpkg_path = out_dir / f"{slug}_prefire_buildings.gpkg"
        gdf_save.to_file(gpkg_path, driver="GPKG")

        save_source_map(gdf_all, perimeter,
                        out_dir / f"{slug}_source_map.png")
        save_spacing_plot(gdf_utm,
                          out_dir / f"{slug}_spacing.png",
                          fire["name"])

        elapsed = time.time() - t0
        src_counts = gdf_all["source"].value_counts().to_dict()

        summary.update({
            "status":           "success",
            "elapsed_sec":      round(elapsed, 1),
            "perimeter_km2":    round(area_km2, 1),
            "bbox":             bbox,
            "pre_fire_date":    pre_date,
            "n_buildings_total": len(gdf_all),
            "n_buildings_utm":  len(gdf_utm),
            "source_counts":    src_counts,
            "ww_median_m":      round(float(np.median(ww)), 1),
            "ww_p10_m":         round(float(np.percentile(ww, 10)), 1),
            "ww_lt3m_pct":      round(float((ww < 3).mean() * 100), 1),
            "ww_lt15m_pct":     round(float((ww < 15).mean() * 100), 1),
            "dins_available":   dins_avail,
            "dins_match_pct":   round(match_pct, 1) if match_pct else None,
            "output_gpkg":      str(gpkg_path),
            "output_unified":   str(unified_path) if unified_path else None,
        })

        log.info("  ✓  %s complete in %.0f s  (%d buildings)",
                 fire["name"], elapsed, len(gdf_all))

    except Exception as exc:
        log.error("  ✗  %s FAILED: %s", fire["name"], exc, exc_info=True)
        summary["error"] = str(exc)

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


# ═════════════════════════════════════════════════════════════════════
# BATCH RUNNER + SUMMARY
# ═════════════════════════════════════════════════════════════════════

def run_batch(fire_names: list = None, resume: bool = False):
    fires = FIRE_INVENTORY
    if fire_names:
        fires = [f for f in fires if f["name"] in fire_names]
        if not fires:
            log.error("No matching fires found. Available: %s",
                      [f["name"] for f in FIRE_INVENTORY])
            return

    log.info("Running %d fire(s)  |  output_root=%s", len(fires), OUTPUT_ROOT)
    log.info("DINS directory: %s", DINS_DIR)

    summaries = []
    for fire in fires:
        s = process_fire(fire, OUTPUT_ROOT, DINS_DIR, resume=resume)
        summaries.append(s)

    # ── Write batch summary CSV ───────────────────────────────────────
    df_summary = pd.DataFrame(summaries)
    csv_path   = OUTPUT_ROOT / "batch_summary.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(csv_path, index=False)
    log.info("")
    log.info("═" * 60)
    log.info("BATCH COMPLETE")
    log.info("═" * 60)

    success = df_summary[df_summary["status"] == "success"]
    failed  = df_summary[df_summary["status"] != "success"]

    log.info("  Completed : %d / %d fires", len(success), len(fires))
    if len(success):
        log.info("  %-25s %8s  %12s  %10s  %8s",
                 "Fire", "Bldgs", "WW median (m)", "DINS match%", "Elapsed s")
        log.info("  " + "-" * 65)
        for _, r in success.iterrows():
            dins_str = f"{r['dins_match_pct']:.1f}%" if pd.notna(r.get("dins_match_pct")) else "N/A"
            log.info("  %-25s %8d  %12.1f  %10s  %8.0f",
                     r["fire"], r.get("n_buildings_total", 0),
                     r.get("ww_median_m", 0), dins_str, r.get("elapsed_sec", 0))
    if len(failed):
        log.info("  FAILED: %s", list(failed["fire"]))

    log.info("  Summary CSV → %s", csv_path)


# ═════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch pre-fire building footprint compiler for CA WUI fires."
    )
    parser.add_argument(
        "--fires", nargs="*", metavar="NAME",
        help="Fire name(s) to process (e.g. Woolsey Kincade). Default: all."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip fires whose summary.json already exists."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available fire names and exit."
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable fires:")
        for f in FIRE_INVENTORY:
            dins_exists = (DINS_DIR / f"DINS_{f['dins']}.parquet").exists()
            print(f"  {f['name']:<30} {f['date']}  DINS={'✓' if dins_exists else '—'}")
        print()
    else:
        run_batch(fire_names=args.fires, resume=args.resume)
