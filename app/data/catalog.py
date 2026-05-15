"""
Dataset catalog: metadata + sample data generator.

Each dataset entry describes the structure, tags, and schema of the dataset.
Sample data is generated deterministically (seed=42) using NumPy/Pandas.
CSVs are written to app/data/sample_data/ on first run.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


# ---------------------------------------------------------------------------
# Dataset Metadata Catalog
# ---------------------------------------------------------------------------

DATASET_CATALOG: List[Dict[str, Any]] = [
    {
        "dataset_id": "air_quality",
        "dataset_name_en": "Air Quality Dataset",
        "dataset_name_ar": "مجموعة بيانات جودة الهواء",
        "description": "Monthly air quality measurements (PM2.5, PM10, NO2, CO, AQI) across major Saudi cities from 2020 to 2024.",
        "columns": ["date", "city", "region", "pm25", "pm10", "no2", "co", "aqi_value", "aqi_category"],
        "date_column": "date",
        "geography_column": "region",
        "numeric_columns": ["pm25", "pm10", "no2", "co", "aqi_value"],
        "categorical_columns": ["city", "region", "aqi_category"],
        "tags": ["air", "pollution", "aqi", "pm25", "health", "monitoring"],
        "source": "Ministry of Environment — Mock Sample Data",
        "quality_score": 0.87,
    },
    {
        "dataset_id": "water_consumption",
        "dataset_name_en": "Water Consumption Dataset",
        "dataset_name_ar": "مجموعة بيانات استهلاك المياه",
        "description": "Monthly water consumption by region and sector (residential, industrial, agricultural) from 2020 to 2024.",
        "columns": ["date", "region", "sector", "total_consumption_m3", "population", "per_capita_m3"],
        "date_column": "date",
        "geography_column": "region",
        "numeric_columns": ["total_consumption_m3", "population", "per_capita_m3"],
        "categorical_columns": ["region", "sector"],
        "tags": ["water", "consumption", "scarcity", "region", "sustainability"],
        "source": "National Water Company — Mock Sample Data",
        "quality_score": 0.91,
    },
    {
        "dataset_id": "vegetation_coverage",
        "dataset_name_en": "Vegetation Coverage Dataset",
        "dataset_name_ar": "مجموعة بيانات الغطاء النباتي",
        "description": "Annual vegetation coverage percentage and NDVI values by region from 2015 to 2024.",
        "columns": ["year", "region", "coverage_pct", "area_km2", "ndvi_index", "vegetation_type"],
        "date_column": "year",
        "geography_column": "region",
        "numeric_columns": ["coverage_pct", "area_km2", "ndvi_index"],
        "categorical_columns": ["region", "vegetation_type"],
        "tags": ["vegetation", "greenery", "ndvi", "land", "coverage", "desertification"],
        "source": "NCVC Satellite Data — Mock Sample Data",
        "quality_score": 0.83,
    },
    {
        "dataset_id": "environmental_violations",
        "dataset_name_en": "Environmental Violations Dataset",
        "dataset_name_ar": "مجموعة بيانات المخالفات البيئية",
        "description": "Environmental violation records including region, category, severity, and resolution status from 2020 to 2024.",
        "columns": ["violation_id", "date", "region", "category", "severity", "status", "fine_sar"],
        "date_column": "date",
        "geography_column": "region",
        "numeric_columns": ["fine_sar"],
        "categorical_columns": ["region", "category", "severity", "status"],
        "tags": ["violations", "enforcement", "region", "penalty", "compliance", "legal"],
        "source": "Ministry of Environment — Mock Sample Data",
        "quality_score": 0.89,
    },
    {
        "dataset_id": "waste_management",
        "dataset_name_en": "Waste Management Dataset",
        "dataset_name_ar": "مجموعة بيانات إدارة النفايات",
        "description": "Monthly waste collection and recycling statistics by region from 2020 to 2024.",
        "columns": ["date", "region", "total_waste_tons", "recycled_tons", "recycling_rate", "waste_type"],
        "date_column": "date",
        "geography_column": "region",
        "numeric_columns": ["total_waste_tons", "recycled_tons", "recycling_rate"],
        "categorical_columns": ["region", "waste_type"],
        "tags": ["waste", "recycling", "sustainability", "region", "circular economy"],
        "source": "MEWA — Mock Sample Data",
        "quality_score": 0.85,
    },
    {
        "dataset_id": "protected_areas",
        "dataset_name_en": "Protected Areas Dataset",
        "dataset_name_ar": "مجموعة بيانات المناطق المحمية",
        "description": "Registered environmental protected areas across Saudi Arabia including size, type, and establishment year.",
        "columns": ["area_id", "name_en", "name_ar", "region", "area_km2", "year_established", "type", "species_count"],
        "date_column": "year_established",
        "geography_column": "region",
        "numeric_columns": ["area_km2", "species_count"],
        "categorical_columns": ["region", "type"],
        "tags": ["protected areas", "biodiversity", "nature reserves", "conservation"],
        "source": "Saudi Wildlife Authority — Mock Sample Data",
        "quality_score": 0.93,
    },
    {
        "dataset_id": "climate_indicators",
        "dataset_name_en": "Climate Indicators Dataset",
        "dataset_name_ar": "مجموعة بيانات المؤشرات المناخية",
        "description": "Monthly climate data including temperature, rainfall, humidity, and dust events by region from 2018 to 2024.",
        "columns": ["date", "region", "avg_temp_c", "max_temp_c", "rainfall_mm", "humidity_pct", "dust_days"],
        "date_column": "date",
        "geography_column": "region",
        "numeric_columns": ["avg_temp_c", "max_temp_c", "rainfall_mm", "humidity_pct", "dust_days"],
        "categorical_columns": ["region"],
        "tags": ["climate", "temperature", "rainfall", "humidity", "dust", "weather"],
        "source": "Saudi Meteorological Authority — Mock Sample Data",
        "quality_score": 0.90,
    },
]

# Quick lookup by ID
CATALOG_BY_ID: Dict[str, Dict[str, Any]] = {d["dataset_id"]: d for d in DATASET_CATALOG}


# ---------------------------------------------------------------------------
# Sample Data Generators
# ---------------------------------------------------------------------------

REGIONS = ["Riyadh", "Makkah", "Eastern Province", "Madinah", "Asir", "Tabuk", "Qassim", "Jizan"]
REGION_WEIGHTS = [0.25, 0.20, 0.18, 0.12, 0.08, 0.07, 0.06, 0.04]


def _make_dates(start: str, end: str, n: int, sort: bool = True) -> List[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days
    offsets = np.random.randint(0, delta_days, size=n)
    dates = [start_dt + timedelta(days=int(d)) for d in offsets]
    if sort:
        dates.sort()
    return [d.strftime("%Y-%m-%d") for d in dates]


def generate_violations(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = 500
    dates = _make_dates("2020-01-01", "2024-12-31", n)
    categories = ["Air Pollution", "Water Discharge", "Waste Dumping", "Industrial Emission", "Noise Pollution", "Land Degradation"]
    severities = ["Low", "Medium", "High", "Critical"]
    statuses = ["Open", "Closed", "Under Review", "Resolved"]
    return pd.DataFrame({
        "violation_id": [f"V{i:05d}" for i in range(1, n + 1)],
        "date": dates,
        "region": np.random.choice(REGIONS, n, p=REGION_WEIGHTS),
        "category": np.random.choice(categories, n, p=[0.28, 0.20, 0.18, 0.15, 0.10, 0.09]),
        "severity": np.random.choice(severities, n, p=[0.40, 0.30, 0.20, 0.10]),
        "status": np.random.choice(statuses, n, p=[0.25, 0.45, 0.15, 0.15]),
        "fine_sar": np.random.choice([0, 5000, 10000, 25000, 50000, 100000], n,
                                     p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]),
    })


def generate_air_quality(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 1)
    cities = {
        "Riyadh": "Riyadh", "Jeddah": "Makkah", "Makkah": "Makkah",
        "Dammam": "Eastern Province", "Khobar": "Eastern Province",
        "Madinah": "Madinah", "Abha": "Asir", "Tabuk": "Tabuk",
    }
    rows = []
    for year in range(2020, 2025):
        for month in range(1, 13):
            for city, region in cities.items():
                base_pm25 = 35 if region in ["Riyadh", "Eastern Province"] else 20
                rows.append({
                    "date": f"{year}-{month:02d}-01",
                    "city": city,
                    "region": region,
                    "pm25": round(base_pm25 + np.random.normal(0, 8), 1),
                    "pm10": round((base_pm25 * 1.8) + np.random.normal(0, 12), 1),
                    "no2": round(np.random.uniform(10, 60), 1),
                    "co": round(np.random.uniform(0.3, 2.5), 2),
                    "aqi_value": int(np.random.randint(40, 180)),
                    "aqi_category": np.random.choice(
                        ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy"],
                        p=[0.25, 0.40, 0.25, 0.10]
                    ),
                })
    return pd.DataFrame(rows)


def generate_water_consumption(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 2)
    sectors = ["Residential", "Industrial", "Agricultural", "Commercial"]
    rows = []
    region_pop = {
        "Riyadh": 7_500_000, "Makkah": 4_200_000, "Eastern Province": 3_800_000,
        "Madinah": 2_100_000, "Asir": 1_800_000, "Tabuk": 900_000,
        "Qassim": 1_100_000, "Jizan": 1_500_000,
    }
    for year in range(2020, 2025):
        for month in range(1, 13):
            for region, pop in region_pop.items():
                for sector in sectors:
                    base = pop * np.random.uniform(0.004, 0.008)
                    rows.append({
                        "date": f"{year}-{month:02d}-01",
                        "region": region,
                        "sector": sector,
                        "total_consumption_m3": int(base * np.random.uniform(0.9, 1.1)),
                        "population": pop,
                        "per_capita_m3": round(base / pop, 3),
                    })
    return pd.DataFrame(rows)


def generate_vegetation_coverage(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 3)
    vegetation_types = ["Shrubland", "Grassland", "Forest", "Desert Scrub", "Mangrove"]
    base_coverage = {
        "Riyadh": 1.5, "Makkah": 3.0, "Eastern Province": 2.0,
        "Madinah": 4.0, "Asir": 12.0, "Tabuk": 5.0, "Qassim": 3.5, "Jizan": 8.0,
    }
    rows = []
    for year in range(2015, 2025):
        for region, base in base_coverage.items():
            trend = 0.05 * (year - 2015)
            cov = round(base + trend + np.random.normal(0, 0.3), 2)
            rows.append({
                "year": year,
                "region": region,
                "coverage_pct": max(0.1, cov),
                "area_km2": round(cov * np.random.uniform(300, 800), 0),
                "ndvi_index": round(np.random.uniform(0.05, 0.45), 3),
                "vegetation_type": np.random.choice(vegetation_types),
            })
    return pd.DataFrame(rows)


def generate_waste_management(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 4)
    waste_types = ["Municipal Solid Waste", "Industrial Waste", "Construction Debris", "Hazardous Waste"]
    rows = []
    region_scale = {
        "Riyadh": 9000, "Makkah": 6000, "Eastern Province": 5500,
        "Madinah": 3000, "Asir": 2500, "Tabuk": 1200, "Qassim": 1500, "Jizan": 2000,
    }
    for year in range(2020, 2025):
        for month in range(1, 13):
            for region, scale in region_scale.items():
                total = int(scale * np.random.uniform(0.85, 1.15))
                rate = round(np.random.uniform(0.05, 0.25), 3)
                rows.append({
                    "date": f"{year}-{month:02d}-01",
                    "region": region,
                    "total_waste_tons": total,
                    "recycled_tons": int(total * rate),
                    "recycling_rate": rate,
                    "waste_type": np.random.choice(waste_types),
                })
    return pd.DataFrame(rows)


def generate_protected_areas(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 5)
    areas = [
        ("Asir National Park", "منتزه عسير الوطني", "Asir", 4500, 1981, "National Park", 285),
        ("Harrat Khaybar", "حرة خيبر", "Madinah", 12300, 1995, "Nature Reserve", 120),
        ("Uruq Bani Maarid", "عروق بني معارض", "Riyadh", 12000, 1994, "Nature Reserve", 205),
        ("Al-Jubail Marine Wildlife Sanctuary", "محمية الجبيل البحرية", "Eastern Province", 820, 2001, "Marine Sanctuary", 310),
        ("Farasan Islands", "جزر فرسان", "Jizan", 840, 2004, "Marine Protected Area", 450),
        ("Mahazat as-Sayd", "محاظة الصيد", "Makkah", 2244, 1988, "Nature Reserve", 98),
        ("Al-Khunfah", "الخنفة", "Tabuk", 5700, 2000, "Nature Reserve", 75),
        ("Raydah", "ريدة", "Asir", 21, 1987, "National Park", 165),
        ("Al-Tubayq", "التبيق", "Tabuk", 12000, 1988, "Nature Reserve", 112),
        ("Ibex Reserve", "محمية الوعل", "Riyadh", 1600, 1987, "Nature Reserve", 88),
    ]
    rows = []
    for i, (name_en, name_ar, region, area_km2, year_est, ptype, species) in enumerate(areas, 1):
        rows.append({
            "area_id": f"PA{i:03d}",
            "name_en": name_en,
            "name_ar": name_ar,
            "region": region,
            "area_km2": area_km2,
            "year_established": year_est,
            "type": ptype,
            "species_count": species,
        })
    return pd.DataFrame(rows)


def generate_climate_indicators(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed + 6)
    region_temps = {
        "Riyadh": 31, "Makkah": 33, "Eastern Province": 29,
        "Madinah": 30, "Asir": 22, "Tabuk": 28, "Qassim": 30, "Jizan": 32,
    }
    rows = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            for region, base_temp in region_temps.items():
                seasonal = -8 * np.cos(2 * np.pi * (month - 7) / 12)
                rows.append({
                    "date": f"{year}-{month:02d}-01",
                    "region": region,
                    "avg_temp_c": round(base_temp + seasonal + np.random.normal(0, 1.5), 1),
                    "max_temp_c": round(base_temp + seasonal + 8 + np.random.normal(0, 2), 1),
                    "rainfall_mm": round(max(0, np.random.exponential(5 if month in [1, 2, 11, 12] else 1)), 1),
                    "humidity_pct": round(np.random.uniform(20, 75), 1),
                    "dust_days": int(np.random.randint(0, 8 if month in [3, 4, 5] else 3)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_GENERATORS = {
    "environmental_violations": generate_violations,
    "air_quality": generate_air_quality,
    "water_consumption": generate_water_consumption,
    "vegetation_coverage": generate_vegetation_coverage,
    "waste_management": generate_waste_management,
    "protected_areas": generate_protected_areas,
    "climate_indicators": generate_climate_indicators,
}

_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Load a dataset by ID. Uses in-memory cache.
    Generates from scratch on first call and saves to CSV.
    """
    if dataset_id in _DATA_CACHE:
        return _DATA_CACHE[dataset_id].copy()

    csv_path = SAMPLE_DATA_DIR / f"{dataset_id}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if dataset_id not in _GENERATORS:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        logger.info(f"Generating sample data for: {dataset_id}")
        df = _GENERATORS[dataset_id]()
        SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved sample data to {csv_path}")

    _DATA_CACHE[dataset_id] = df
    return df.copy()


def get_catalog_for_llm() -> str:
    """
    Returns a compact text representation of all datasets
    for inclusion in LLM system prompts.
    """
    lines = []
    for d in DATASET_CATALOG:
        lines.append(f"- ID: {d['dataset_id']}")
        lines.append(f"  Name: {d['dataset_name_en']} / {d['dataset_name_ar']}")
        lines.append(f"  Description: {d['description']}")
        lines.append(f"  Columns: {', '.join(d['columns'])}")
        lines.append(f"  Date column: {d['date_column']}")
        lines.append(f"  Numeric columns: {', '.join(d['numeric_columns'])}")
        lines.append(f"  Categorical columns: {', '.join(d['categorical_columns'])}")
        lines.append(f"  Tags: {', '.join(d['tags'])}")
        lines.append("")
    return "\n".join(lines)


def get_dataset_summary(dataset_id: str) -> Dict[str, Any]:
    """Returns a summary dict for a loaded dataset."""
    meta = CATALOG_BY_ID.get(dataset_id, {})
    df = load_dataset(dataset_id)

    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "name_en": meta.get("dataset_name_en", dataset_id),
        "name_ar": meta.get("dataset_name_ar", ""),
        "record_count": len(df),
        "key_stats": {},
    }

    date_col = meta.get("date_column")
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                summary["date_range"] = f"{dates.min().year}–{dates.max().year}"
        except Exception:
            pass

    geo_col = meta.get("geography_column")
    if geo_col and geo_col in df.columns:
        summary["key_stats"]["Regions covered"] = ", ".join(df[geo_col].unique().tolist()[:6])

    for cat_col in meta.get("categorical_columns", [])[:2]:
        if cat_col in df.columns and cat_col != geo_col:
            top = df[cat_col].value_counts().head(3).to_dict()
            summary["key_stats"][f"Top {cat_col}"] = ", ".join(f"{k} ({v})" for k, v in top.items())

    return summary


def initialize_all_datasets() -> None:
    """Pre-generate and cache all datasets at startup."""
    for dataset_id in _GENERATORS:
        try:
            load_dataset(dataset_id)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
