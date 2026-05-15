from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# KPI Plan — what the LLM wants to display as a KPI card
# ---------------------------------------------------------------------------

ALLOWED_CALCULATIONS = Literal[
    "count_rows",
    "sum",
    "mean",
    "max",
    "min",
    "top_category_by_count",
    "count_unique",
    "latest_value",
    "pct_change",
]


class KPIPlan(BaseModel):
    id: str
    label: str
    label_ar: Optional[str] = None
    calculation: ALLOWED_CALCULATIONS
    column: str
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Chart Spec — one chart in the dashboard
# ---------------------------------------------------------------------------

ALLOWED_CHART_TYPES = Literal["line", "bar", "donut", "scatter", "heatmap"]
ALLOWED_AGGREGATIONS = Literal["count", "sum", "mean", "max", "min"]


class ChartSpec(BaseModel):
    id: str
    type: ALLOWED_CHART_TYPES
    title: str
    x: str
    y: str
    aggregation: ALLOWED_AGGREGATIONS
    color_by: Optional[str] = None
    reason: str


# ---------------------------------------------------------------------------
# Insight Hint — guidance the LLM provides for generating a written insight
# ---------------------------------------------------------------------------

class InsightHint(BaseModel):
    id: str
    type: Literal["trend", "comparison", "anomaly", "recommendation", "summary"]
    source_chart: Optional[str] = None
    description_hint: str


# ---------------------------------------------------------------------------
# Full Dashboard Spec — structured output from the Dashboard Generation Agent
# ---------------------------------------------------------------------------

class DashboardSpec(BaseModel):
    dashboard_title: str
    dashboard_title_ar: Optional[str] = None
    intent: str
    selected_datasets: List[str] = Field(min_length=1)
    kpis: List[KPIPlan] = Field(min_length=1, max_length=4)
    charts: List[ChartSpec] = Field(min_length=1, max_length=2)
    insight_hints: List[InsightHint] = Field(min_length=2, max_length=4)
    assumptions: List[str] = Field(default_factory=list)
    data_limitations: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None

    @field_validator("selected_datasets")
    @classmethod
    def validate_dataset_ids(cls, v: List[str]) -> List[str]:
        allowed = {
            "air_quality", "water_consumption", "vegetation_coverage",
            "environmental_violations", "waste_management", "protected_areas",
            "climate_indicators",
        }
        for ds_id in v:
            if ds_id not in allowed:
                raise ValueError(f"Unknown dataset: {ds_id}")
        return v

    @field_validator("charts")
    @classmethod
    def validate_chart_count(cls, v: List[ChartSpec]) -> List[ChartSpec]:
        if len(v) > 2:
            return v[:2]
        return v


# ---------------------------------------------------------------------------
# Dataset Recommendation — output from the initial request interpretation
# ---------------------------------------------------------------------------

class DatasetRecommendation(BaseModel):
    dataset_id: str
    dataset_name_en: str
    dataset_name_ar: str
    relevance_reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class InterpretResult(BaseModel):
    interpreted_intent: str
    recommended_datasets: List[DatasetRecommendation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Analyst response — output from the Business Analyst Agent
# ---------------------------------------------------------------------------

class AnalystResponse(BaseModel):
    answer: str
    supporting_metrics: List[str] = Field(default_factory=list)
    scope_status: Literal["in_scope", "out_of_scope"] = "in_scope"
    dashboard_version_used: int = 1
