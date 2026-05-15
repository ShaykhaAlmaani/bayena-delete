"""
Dashboard generator: orchestrates the full dashboard creation pipeline.

Flow:
  1. interpret_request  → LLM (Dashboard Generation Agent) → dataset recommendations
  2. generate_dashboard → LLM spec → validate → compute KPIs → build charts → generate insights
  3. edit_dashboard     → LLM update spec → re-generate dashboard
  4. build_analyst_context → build text context for the Business Analyst Agent
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.config import settings
from app.data.catalog import (
    CATALOG_BY_ID, get_catalog_for_llm, get_dataset_summary, load_dataset
)
from app.llm.client import LLMError, get_llm_client
from app.llm.prompts import (
    ANALYST_SYSTEM_PROMPT,
    DASHBOARD_GENERATION_SYSTEM_PROMPT,
    INTERPRET_REQUEST_SYSTEM_PROMPT,
    build_analyst_context_text,
    build_analyst_prompt,
    build_dashboard_edit_prompt,
    build_dashboard_generation_prompt,
    build_interpret_prompt,
)
from app.llm.structured_outputs import DashboardSpec, InterpretResult
from app.core.kpi_calculator import compute_all_kpis
from app.core.chart_factory import build_chart, build_chart_summary
from app.core.insight_engine import generate_insights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec validation helpers
# ---------------------------------------------------------------------------

def _validate_spec_columns(spec: DashboardSpec, datasets: Dict[str, Any]) -> List[str]:
    """
    Returns a list of warning strings for columns referenced in the spec
    that do not exist in the loaded datasets.
    The generator will attempt to fix them silently.
    """
    warnings = []
    for ds_id in spec.selected_datasets:
        meta = datasets.get(ds_id)
        if meta is None:
            warnings.append(f"Dataset '{ds_id}' is not in the catalog.")
            continue
        available_cols = set(meta.get("columns", []))
        for kpi in spec.kpis:
            if kpi.column not in available_cols:
                warnings.append(f"KPI '{kpi.id}': column '{kpi.column}' not in {ds_id}.")
        for chart in spec.charts:
            if chart.x not in available_cols:
                warnings.append(f"Chart '{chart.id}': x column '{chart.x}' not in {ds_id}.")
            if chart.y not in available_cols and chart.y != "count":
                warnings.append(f"Chart '{chart.id}': y column '{chart.y}' not in {ds_id}.")
    return warnings


def _fix_spec_columns(spec: DashboardSpec, datasets: Dict[str, Any]) -> DashboardSpec:
    """
    Attempt to auto-repair column references that do not exist in the dataset.
    Substitutes with the closest available column from the schema.
    """
    spec_dict = spec.model_dump()

    for ds_id in spec.selected_datasets:
        meta = datasets.get(ds_id)
        if not meta:
            continue
        available = set(meta.get("columns", []))
        numeric = set(meta.get("numeric_columns", []))
        categorical = set(meta.get("categorical_columns", []))
        date_col = meta.get("date_column", "date")

        for kpi in spec_dict["kpis"]:
            if kpi["column"] not in available:
                # Pick first available column of the right type
                fallback = next(iter(categorical), next(iter(numeric), next(iter(available), kpi["column"])))
                logger.warning(f"Auto-fixing KPI column '{kpi['column']}' → '{fallback}'")
                kpi["column"] = fallback

        for chart in spec_dict["charts"]:
            if chart["x"] not in available:
                fallback = date_col if date_col in available else next(iter(categorical), chart["x"])
                logger.warning(f"Auto-fixing chart x '{chart['x']}' → '{fallback}'")
                chart["x"] = fallback
            if chart["y"] not in available and chart["y"] != "count":
                fallback = next(iter(numeric), "count")
                logger.warning(f"Auto-fixing chart y '{chart['y']}' → '{fallback}'")
                chart["y"] = fallback

    return DashboardSpec(**spec_dict)


# ---------------------------------------------------------------------------
# Methodology text builder
# ---------------------------------------------------------------------------

def _build_methodology(
    spec: DashboardSpec,
    dataset_ids: List[str],
    is_mock: bool,
) -> str:
    ds_names = [CATALOG_BY_ID.get(d, {}).get("dataset_name_en", d) for d in dataset_ids]
    chart_reasons = " ".join(f"• {c.title} — {c.reason}" for c in spec.charts)
    limitations = "; ".join(spec.data_limitations) if spec.data_limitations else "None specified."
    data_note = "⚠ Sample/mock data — not real government records." if is_mock else "Live dataset."
    return (
        f"Datasets used: {', '.join(ds_names)}. "
        f"Chart selection: {chart_reasons} "
        f"Assumptions: {'; '.join(spec.assumptions) if spec.assumptions else 'None.'}  "
        f"Limitations: {limitations}  "
        f"Data source note: {data_note}"
    )


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------

def interpret_request(user_prompt: str) -> Dict[str, Any]:
    """
    Step 1: Use the Dashboard Generation Agent to recommend datasets.
    Returns a dict matching the InterpretResult schema.
    """
    try:
        client = get_llm_client("dashboard")
        catalog_text = get_catalog_for_llm()
        system_prompt = INTERPRET_REQUEST_SYSTEM_PROMPT.replace("{dataset_catalog}", catalog_text)
        user_message = build_interpret_prompt(user_prompt)
        raw = client.chat_json(system_prompt, user_message, temperature=0.1)
        result = InterpretResult(**raw)
        return result.model_dump()
    except (LLMError, ValidationError, Exception) as e:
        logger.error(f"interpret_request failed: {e}")
        # Graceful fallback
        return {
            "interpreted_intent": f"Analyze: {user_prompt[:100]}",
            "recommended_datasets": [
                {
                    "dataset_id": "environmental_violations",
                    "dataset_name_en": "Environmental Violations Dataset",
                    "dataset_name_ar": "مجموعة بيانات المخالفات البيئية",
                    "relevance_reason": "Default dataset selected due to interpretation error.",
                    "confidence": 0.5,
                }
            ],
            "confidence_score": 0.5,
            "clarification_needed": False,
            "clarification_question": None,
        }


def generate_dashboard(
    user_prompt: str,
    selected_dataset_ids: List[str],
    retry_on_invalid: bool = True,
) -> Dict[str, Any]:
    """
    Step 2: Generate a full dashboard from the selected datasets.

    Returns a dict with: dashboard_spec, kpi_cards, charts (Plotly dicts),
    insights, methodology, dataset_summaries.
    """
    # Validate dataset IDs
    valid_ids = [d for d in selected_dataset_ids if d in CATALOG_BY_ID]
    if not valid_ids:
        raise ValueError("None of the selected dataset IDs are in the catalog.")

    catalog_text = get_catalog_for_llm()
    client = get_llm_client("dashboard")
    system_prompt = DASHBOARD_GENERATION_SYSTEM_PROMPT.replace("{dataset_catalog}", catalog_text)
    user_message = build_dashboard_generation_prompt(user_prompt, catalog_text)

    # Get spec from LLM
    try:
        raw = client.chat_json(system_prompt, user_message, temperature=0.1)
        spec = DashboardSpec(**raw)
    except (LLMError, ValidationError, Exception) as e:
        logger.error(f"LLM spec generation failed: {e}")
        if retry_on_invalid:
            logger.info("Falling back to mock spec.")
            from app.llm.client import MockLLMClient
            mock = MockLLMClient()
            raw = mock.chat_json(system_prompt, user_message)
            spec = DashboardSpec(**raw)
        else:
            raise

    # Override selected datasets with what user confirmed
    spec_dict = spec.model_dump()
    spec_dict["selected_datasets"] = valid_ids
    spec = DashboardSpec(**spec_dict)

    # Validate and fix column references
    warnings = _validate_spec_columns(spec, CATALOG_BY_ID)
    if warnings:
        logger.warning(f"Column validation warnings: {warnings}")
        spec = _fix_spec_columns(spec, CATALOG_BY_ID)

    # Load primary dataset (use first selected dataset)
    primary_ds_id = spec.selected_datasets[0]
    df = load_dataset(primary_ds_id)

    # Compute KPIs
    kpi_cards = compute_all_kpis(spec.kpis, df)

    # Build charts
    charts = []
    chart_metas = []
    for chart_spec in spec.charts:
        try:
            fig_dict = build_chart(chart_spec, df)
            summary = build_chart_summary(chart_spec, df)
            charts.append({
                "id": chart_spec.id,
                "title": chart_spec.title,
                "type": chart_spec.type,
                "figure": fig_dict,
                "reason": chart_spec.reason,
                "summary": summary,
            })
            chart_metas.append({
                "id": chart_spec.id,
                "title": chart_spec.title,
                "type": chart_spec.type,
                "summary": summary,
            })
        except ValueError as e:
            logger.error(f"Chart build error for '{chart_spec.id}': {e}")
            charts.append({
                "id": chart_spec.id,
                "title": chart_spec.title,
                "type": chart_spec.type,
                "figure": None,
                "error": str(e),
                "reason": chart_spec.reason,
            })
            chart_metas.append({
                "id": chart_spec.id,
                "title": chart_spec.title,
                "type": chart_spec.type,
                "summary": f"Chart could not be rendered: {e}",
            })

    # Generate insights
    chart_spec_dicts = [c.model_dump() for c in spec.charts]
    insights = generate_insights(spec.insight_hints, df, chart_spec_dicts, kpi_cards)

    # Dataset summaries
    dataset_summaries = [get_dataset_summary(ds_id) for ds_id in valid_ids]

    # Methodology
    is_mock = not settings.GROQ_API_KEY
    methodology = _build_methodology(spec, valid_ids, is_mock)

    return {
        "dashboard_title": spec.dashboard_title,
        "dashboard_title_ar": spec.dashboard_title_ar,
        "intent": spec.intent,
        "dashboard_spec": spec.model_dump(),
        "kpi_cards": kpi_cards,
        "charts": charts,
        "insights": insights,
        "methodology": methodology,
        "dataset_summaries": dataset_summaries,
        "data_limitations": spec.data_limitations,
        "assumptions": spec.assumptions,
        "confidence_score": spec.confidence_score,
    }


def edit_dashboard(
    user_prompt: str,
    edit_request: str,
    current_spec: Dict[str, Any],
    selected_dataset_ids: List[str],
) -> Dict[str, Any]:
    """
    Step 3: Edit an existing dashboard based on user feedback.
    Re-invokes the Dashboard Generation Agent with the current spec + edit request.
    """
    catalog_text = get_catalog_for_llm()
    client = get_llm_client("dashboard")
    system_prompt = DASHBOARD_GENERATION_SYSTEM_PROMPT.replace("{dataset_catalog}", catalog_text)
    user_message = build_dashboard_edit_prompt(
        original_prompt=user_prompt,
        edit_request=edit_request,
        current_spec_json=json.dumps(current_spec, indent=2),
        dataset_catalog_text=catalog_text,
    )

    try:
        raw = client.chat_json(system_prompt, user_message, temperature=0.1)
        spec = DashboardSpec(**raw)
    except (LLMError, ValidationError, Exception) as e:
        logger.error(f"Edit dashboard spec failed: {e}")
        # Fall back to regenerating from scratch
        return generate_dashboard(f"{user_prompt}. User edit: {edit_request}", selected_dataset_ids)

    spec_dict = spec.model_dump()
    spec_dict["selected_datasets"] = [d for d in selected_dataset_ids if d in CATALOG_BY_ID] or spec.selected_datasets
    spec = DashboardSpec(**spec_dict)
    spec = _fix_spec_columns(spec, CATALOG_BY_ID)

    primary_ds_id = spec.selected_datasets[0]
    df = load_dataset(primary_ds_id)

    kpi_cards = compute_all_kpis(spec.kpis, df)

    charts = []
    chart_metas = []
    for chart_spec in spec.charts:
        try:
            fig_dict = build_chart(chart_spec, df)
            summary = build_chart_summary(chart_spec, df)
            charts.append({
                "id": chart_spec.id, "title": chart_spec.title, "type": chart_spec.type,
                "figure": fig_dict, "reason": chart_spec.reason, "summary": summary,
            })
            chart_metas.append({"id": chart_spec.id, "title": chart_spec.title, "type": chart_spec.type, "summary": summary})
        except ValueError as e:
            charts.append({"id": chart_spec.id, "title": chart_spec.title, "type": chart_spec.type, "figure": None, "error": str(e), "reason": chart_spec.reason})
            chart_metas.append({"id": chart_spec.id, "title": chart_spec.title, "type": chart_spec.type, "summary": str(e)})

    chart_spec_dicts = [c.model_dump() for c in spec.charts]
    insights = generate_insights(spec.insight_hints, df, chart_spec_dicts, kpi_cards)
    dataset_summaries = [get_dataset_summary(ds_id) for ds_id in spec.selected_datasets]
    is_mock = not settings.GROQ_API_KEY
    methodology = _build_methodology(spec, spec.selected_datasets, is_mock)

    return {
        "dashboard_title": spec.dashboard_title,
        "dashboard_title_ar": spec.dashboard_title_ar,
        "intent": spec.intent,
        "dashboard_spec": spec.model_dump(),
        "kpi_cards": kpi_cards,
        "charts": charts,
        "insights": insights,
        "methodology": methodology,
        "dataset_summaries": dataset_summaries,
        "data_limitations": spec.data_limitations,
        "assumptions": spec.assumptions,
        "confidence_score": spec.confidence_score,
    }


def build_analyst_context(
    dashboard_title: str,
    intent: str,
    kpi_cards: List[Dict[str, Any]],
    charts: List[Dict[str, Any]],
    insights: List[str],
    dataset_summaries: List[Dict[str, Any]],
    data_limitations: List[str],
) -> str:
    """Build the context string for the Business Analyst Agent."""
    chart_summaries = [
        {"title": c.get("title", ""), "type": c.get("type", ""), "summary": c.get("summary", "")}
        for c in charts
    ]
    return build_analyst_context_text(
        dashboard_title=dashboard_title,
        intent=intent,
        kpi_cards=kpi_cards,
        chart_summaries=chart_summaries,
        insights=insights,
        dataset_summaries=dataset_summaries,
        data_limitations=data_limitations,
    )


def answer_analyst_question(confirmed_context: str, user_question: str) -> str:
    """
    Step 5: Answer a follow-up question using the Business Analyst Agent.
    Only called after dashboard confirmation.
    """
    client = get_llm_client("analyst")
    system_prompt = ANALYST_SYSTEM_PROMPT.replace("{dashboard_context}", confirmed_context)
    user_message = build_analyst_prompt(user_question)

    try:
        return client.chat_text(system_prompt, user_message, temperature=0.2)
    except LLMError as e:
        logger.error(f"Analyst chat failed: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment."
