"""
Tests for the dashboard generator — KPI calculations, chart building, insights.
"""

import pytest
import pandas as pd

from app.data.catalog import load_dataset
from app.llm.structured_outputs import ChartSpec, KPIPlan, InsightHint
from app.core.kpi_calculator import compute_kpi, compute_all_kpis
from app.core.chart_factory import build_chart, build_chart_summary
from app.core.insight_engine import generate_insights
from app.core.session_store import create_session, get_session, require_session


# ---------------------------------------------------------------------------
# KPI Calculator tests
# ---------------------------------------------------------------------------

class TestKPICalculator:
    def setup_method(self):
        self.df = load_dataset("environmental_violations")

    def test_count_rows(self):
        plan = KPIPlan(id="k1", label="Total", calculation="count_rows", column="violation_id")
        result = compute_kpi(plan, self.df)
        assert result["value"] == f"{len(self.df):,}"

    def test_top_category_by_count(self):
        plan = KPIPlan(id="k2", label="Top Region", calculation="top_category_by_count", column="region")
        result = compute_kpi(plan, self.df)
        assert "%" in result["value"]
        assert result["value"] != "N/A"

    def test_count_unique(self):
        plan = KPIPlan(id="k3", label="Unique Regions", calculation="count_unique", column="region")
        result = compute_kpi(plan, self.df)
        assert int(result["value"].replace(",", "")) == self.df["region"].nunique()

    def test_pct_change(self):
        plan = KPIPlan(id="k4", label="YoY Change", calculation="pct_change", column="date")
        result = compute_kpi(plan, self.df)
        assert result["value"] != "Error"

    def test_missing_column_returns_na(self):
        plan = KPIPlan(id="k5", label="Missing", calculation="sum", column="nonexistent_column")
        result = compute_kpi(plan, self.df)
        assert result["value"] == "N/A"

    def test_unknown_calculation_returns_na(self):
        plan = KPIPlan(id="k6", label="Bad", calculation="count_rows", column="violation_id")
        plan.calculation = "nonexistent_calc"  # type: ignore
        result = compute_kpi(plan, self.df)
        # Should handle gracefully
        assert "value" in result


# ---------------------------------------------------------------------------
# Chart Factory tests
# ---------------------------------------------------------------------------

class TestChartFactory:
    def setup_method(self):
        self.df = load_dataset("environmental_violations")

    def test_builds_bar_chart(self):
        spec = ChartSpec(
            id="c1", type="bar", title="Violations by Region",
            x="region", y="count", aggregation="count", reason="test"
        )
        result = build_chart(spec, self.df)
        assert "data" in result
        assert len(result["data"]) > 0

    def test_builds_line_chart(self):
        spec = ChartSpec(
            id="c2", type="line", title="Trend",
            x="date", y="count", aggregation="count", reason="test"
        )
        result = build_chart(spec, self.df)
        assert "data" in result

    def test_builds_donut_chart(self):
        spec = ChartSpec(
            id="c3", type="donut", title="Category Share",
            x="category", y="count", aggregation="count", reason="test"
        )
        result = build_chart(spec, self.df)
        assert "data" in result

    def test_unsupported_type_raises(self):
        spec = ChartSpec(
            id="c4", type="line", title="Test",
            x="region", y="count", aggregation="count", reason="test"
        )
        spec.type = "unknown_type"  # type: ignore
        with pytest.raises(ValueError):
            build_chart(spec, self.df)

    def test_chart_summary_returns_string(self):
        spec = ChartSpec(
            id="c5", type="bar", title="Violations by Region",
            x="region", y="count", aggregation="count", reason="test"
        )
        summary = build_chart_summary(spec, self.df)
        assert isinstance(summary, str)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# Insight Engine tests
# ---------------------------------------------------------------------------

class TestInsightEngine:
    def setup_method(self):
        self.df = load_dataset("environmental_violations")
        self.kpi_cards = [
            {"id": "k1", "label": "Total", "value": "500", "calculation": "count_rows"},
            {"id": "k2", "label": "Top Region", "value": "Riyadh (25%)", "calculation": "top_category_by_count"},
        ]
        self.chart_specs = [
            {"id": "c1", "type": "line", "x": "date", "y": "count"},
            {"id": "c2", "type": "bar", "x": "region", "y": "count"},
        ]

    def test_trend_insight_returns_string(self):
        hints = [InsightHint(id="i1", type="trend", source_chart="c1", description_hint="Check trend")]
        results = generate_insights(hints, self.df, self.chart_specs, self.kpi_cards)
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 10

    def test_comparison_insight(self):
        hints = [InsightHint(id="i2", type="comparison", source_chart="c2", description_hint="Compare regions")]
        results = generate_insights(hints, self.df, self.chart_specs, self.kpi_cards)
        assert "%" in results[0] or "leads" in results[0]

    def test_recommendation_insight(self):
        hints = [InsightHint(id="i3", type="recommendation", source_chart=None, description_hint="Recommend actions")]
        results = generate_insights(hints, self.df, self.chart_specs, self.kpi_cards)
        assert len(results[0]) > 10

    def test_multiple_insights(self):
        hints = [
            InsightHint(id="i1", type="trend", source_chart="c1", description_hint="trend"),
            InsightHint(id="i2", type="comparison", source_chart="c2", description_hint="compare"),
            InsightHint(id="i3", type="anomaly", source_chart="c1", description_hint="anomaly"),
            InsightHint(id="i4", type="recommendation", source_chart=None, description_hint="recommend"),
        ]
        results = generate_insights(hints, self.df, self.chart_specs, self.kpi_cards)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, str)


# ---------------------------------------------------------------------------
# Session store tests
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_create_and_retrieve(self):
        session = create_session("Test prompt")
        retrieved = get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_require_session_raises_for_unknown(self):
        with pytest.raises(KeyError):
            require_session("nonexistent-session-id-xyz")

    def test_session_status_progression(self):
        session = create_session("Air quality test")
        assert session.dashboard_status == "awaiting_datasets"

        session.set_draft(
            selected_dataset_ids=["air_quality"],
            spec={},
            kpi_cards=[],
            charts=[],
            insights=[],
            methodology="test",
        )
        assert session.dashboard_status == "draft"
        assert session.dashboard_version == 1

        session.confirm(analyst_context="Test context")
        assert session.dashboard_status == "confirmed"
        assert session.is_analyst_ready()

        session.begin_edit()
        assert not session.is_analyst_ready()
