"""
Pydantic schemas for all FastAPI request and response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class SuccessResponse(BaseModel):
    status: str = "ok"
    message: str = ""


# ---------------------------------------------------------------------------
# POST /api/dashboard/request
# ---------------------------------------------------------------------------

class DashboardRequestInput(BaseModel):
    user_prompt: str = Field(min_length=5, max_length=2000)


class DatasetRecommendationOut(BaseModel):
    dataset_id: str
    dataset_name_en: str
    dataset_name_ar: str
    relevance_reason: str
    confidence: float


class DashboardRequestOutput(BaseModel):
    session_id: str
    interpreted_intent: str
    recommended_datasets: List[DatasetRecommendationOut]
    confidence_score: float
    clarification_needed: bool
    clarification_question: Optional[str]
    status: str = "awaiting_dataset_confirmation"


# ---------------------------------------------------------------------------
# POST /api/dashboard/generate
# ---------------------------------------------------------------------------

class DashboardGenerateInput(BaseModel):
    session_id: str
    selected_dataset_ids: List[str] = Field(min_length=1)


class KPICardOut(BaseModel):
    id: str
    label: str
    label_ar: str
    value: str
    description: str


class ChartOut(BaseModel):
    id: str
    title: str
    type: str
    figure: Optional[Dict[str, Any]]  # Plotly figure dict
    error: Optional[str] = None
    reason: str
    summary: Optional[str] = None


class DatasetSummaryOut(BaseModel):
    dataset_id: str
    name_en: str
    name_ar: str
    record_count: int
    date_range: Optional[str] = None
    key_stats: Dict[str, str] = {}


class DashboardGenerateOutput(BaseModel):
    session_id: str
    dashboard_id: str          # same as session_id for simplicity
    dashboard_version: int
    dashboard_status: str      # "draft"
    dashboard_title: str
    dashboard_title_ar: Optional[str]
    intent: str
    kpi_cards: List[KPICardOut]
    charts: List[ChartOut]
    insights: List[str]
    methodology: str
    dataset_summaries: List[DatasetSummaryOut]
    data_limitations: List[str]
    analyst_chat_enabled: bool = False


# ---------------------------------------------------------------------------
# POST /api/dashboard/edit
# ---------------------------------------------------------------------------

class DashboardEditInput(BaseModel):
    session_id: str
    user_edit_request: str = Field(min_length=5, max_length=2000)


# DashboardEditOutput is the same structure as DashboardGenerateOutput

# ---------------------------------------------------------------------------
# POST /api/dashboard/confirm
# ---------------------------------------------------------------------------

class DashboardConfirmInput(BaseModel):
    session_id: str


class DashboardConfirmOutput(BaseModel):
    session_id: str
    dashboard_status: str      # "confirmed"
    dashboard_version: int
    analyst_chat_enabled: bool = True
    message: str = "Dashboard confirmed. Analyst chat is now active."


# ---------------------------------------------------------------------------
# POST /api/analyst-chat
# ---------------------------------------------------------------------------

class AnalystChatInput(BaseModel):
    session_id: str
    user_question: str = Field(min_length=2, max_length=1000)


class AnalystChatOutput(BaseModel):
    session_id: str
    answer: str
    scope_status: str          # "in_scope" or "out_of_scope"
    dashboard_version_used: int


# ---------------------------------------------------------------------------
# GET /api/datasets
# ---------------------------------------------------------------------------

class DatasetCatalogItemOut(BaseModel):
    dataset_id: str
    dataset_name_en: str
    dataset_name_ar: str
    description: str
    columns: List[str]
    tags: List[str]
    source: str
    quality_score: float


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
