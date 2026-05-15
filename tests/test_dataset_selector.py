"""
Tests for the dataset selector — catalog loading and column validation.
"""

import pytest
from app.data.catalog import (
    CATALOG_BY_ID,
    DATASET_CATALOG,
    get_catalog_for_llm,
    get_dataset_summary,
    load_dataset,
)


class TestCatalogMetadata:
    def test_all_expected_datasets_present(self):
        expected = {
            "air_quality", "water_consumption", "vegetation_coverage",
            "environmental_violations", "waste_management", "protected_areas",
            "climate_indicators",
        }
        assert expected == set(CATALOG_BY_ID.keys())

    def test_each_dataset_has_required_fields(self):
        required = {"dataset_id", "dataset_name_en", "dataset_name_ar", "columns",
                    "date_column", "geography_column", "numeric_columns", "categorical_columns"}
        for d in DATASET_CATALOG:
            for field in required:
                assert field in d, f"Dataset '{d['dataset_id']}' missing field '{field}'"

    def test_catalog_for_llm_contains_all_ids(self):
        text = get_catalog_for_llm()
        for d in DATASET_CATALOG:
            assert d["dataset_id"] in text


class TestLoadDataset:
    def test_loads_violations(self):
        df = load_dataset("environmental_violations")
        assert len(df) > 0
        assert "region" in df.columns
        assert "category" in df.columns
        assert "date" in df.columns

    def test_loads_air_quality(self):
        df = load_dataset("air_quality")
        assert "pm25" in df.columns
        assert "aqi_value" in df.columns

    def test_loads_vegetation(self):
        df = load_dataset("vegetation_coverage")
        assert "coverage_pct" in df.columns

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            load_dataset("nonexistent_dataset")

    def test_returns_copy(self):
        df1 = load_dataset("environmental_violations")
        df2 = load_dataset("environmental_violations")
        df1.loc[0, "region"] = "MODIFIED"
        assert df2.loc[0, "region"] != "MODIFIED"


class TestDatasetSummary:
    def test_summary_has_expected_keys(self):
        summary = get_dataset_summary("environmental_violations")
        assert "dataset_id" in summary
        assert "name_en" in summary
        assert "record_count" in summary
        assert summary["record_count"] > 0

    def test_date_range_populated(self):
        summary = get_dataset_summary("environmental_violations")
        assert "date_range" in summary
        assert summary["date_range"] is not None
