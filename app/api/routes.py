"""
FastAPI route handlers.

Two-agent architecture:
  - Dashboard Generation Agent: /api/dashboard/request, /generate, /edit
  - Business Analyst Agent:     /api/analyst-chat
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    AnalystChatInput, AnalystChatOutput,
    DashboardConfirmInput, DashboardConfirmOutput,
    DashboardEditInput, DashboardGenerateInput, DashboardGenerateOutput,
    DashboardRequestInput, DashboardRequestOutput,
    DatasetCatalogItemOut,
)
from app.core.dashboard_generator import (
    answer_analyst_question,
    build_analyst_context,
    edit_dashboard,
    generate_dashboard,
    interpret_request,
)
from app.core.guardrails import (
    check_analyst_question,
    sanitize_user_prompt,
    validate_prompt_length,
)
from app.core.session_store import (
    create_session,
    require_session,
)
from app.data.catalog import DATASET_CATALOG

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "Bayena Dashboard API"}


# ---------------------------------------------------------------------------
# GET /api/datasets — dataset catalog
# ---------------------------------------------------------------------------

@router.get("/datasets", response_model=List[DatasetCatalogItemOut])
def get_datasets():
    return [
        DatasetCatalogItemOut(
            dataset_id=d["dataset_id"],
            dataset_name_en=d["dataset_name_en"],
            dataset_name_ar=d["dataset_name_ar"],
            description=d["description"],
            columns=d["columns"],
            tags=d["tags"],
            source=d["source"],
            quality_score=d["quality_score"],
        )
        for d in DATASET_CATALOG
    ]


# ---------------------------------------------------------------------------
# POST /api/dashboard/request — Step 1: interpret prompt & recommend datasets
# ---------------------------------------------------------------------------

@router.post("/dashboard/request", response_model=DashboardRequestOutput)
def dashboard_request(body: DashboardRequestInput):
    prompt = sanitize_user_prompt(body.user_prompt)
    valid, msg = validate_prompt_length(prompt)
    if not valid:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg)

    session = create_session(prompt)

    try:
        result = interpret_request(prompt)
    except Exception as e:
        logger.error(f"interpret_request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request interpretation failed: {str(e)}")

    session.set_interpreted(
        intent=result["interpreted_intent"],
        recommended_datasets=result["recommended_datasets"],
    )

    return DashboardRequestOutput(
        session_id=session.session_id,
        interpreted_intent=result["interpreted_intent"],
        recommended_datasets=result["recommended_datasets"],
        confidence_score=result["confidence_score"],
        clarification_needed=result["clarification_needed"],
        clarification_question=result.get("clarification_question"),
        status="awaiting_dataset_confirmation",
    )


# ---------------------------------------------------------------------------
# POST /api/dashboard/generate — Step 2: generate draft dashboard
# ---------------------------------------------------------------------------

@router.post("/dashboard/generate", response_model=DashboardGenerateOutput)
def dashboard_generate(body: DashboardGenerateInput):
    session = _get_session_or_404(body.session_id)

    try:
        result = generate_dashboard(
            user_prompt=session.user_prompt,
            selected_dataset_ids=body.selected_dataset_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"dashboard_generate error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

    session.set_draft(
        selected_dataset_ids=body.selected_dataset_ids,
        spec=result["dashboard_spec"],
        kpi_cards=result["kpi_cards"],
        charts=result["charts"],
        insights=result["insights"],
        methodology=result["methodology"],
    )

    return _build_generate_output(session.session_id, session.dashboard_version, "draft", result)


# ---------------------------------------------------------------------------
# POST /api/dashboard/edit — Step 3: edit draft or confirmed dashboard
# ---------------------------------------------------------------------------

@router.post("/dashboard/edit", response_model=DashboardGenerateOutput)
def dashboard_edit(body: DashboardEditInput):
    session = _get_session_or_404(body.session_id)

    if session.dashboard_status not in ("draft", "confirmed", "edited_after_confirmation"):
        raise HTTPException(
            status_code=400,
            detail="No dashboard to edit. Generate a dashboard first.",
        )

    session.begin_edit()

    current_spec = session.draft_spec or {}
    try:
        result = edit_dashboard(
            user_prompt=session.user_prompt,
            edit_request=sanitize_user_prompt(body.user_edit_request),
            current_spec=current_spec,
            selected_dataset_ids=session.selected_dataset_ids,
        )
    except Exception as e:
        logger.error(f"dashboard_edit error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard edit failed: {str(e)}")

    session.set_draft(
        selected_dataset_ids=result["dashboard_spec"].get("selected_datasets", session.selected_dataset_ids),
        spec=result["dashboard_spec"],
        kpi_cards=result["kpi_cards"],
        charts=result["charts"],
        insights=result["insights"],
        methodology=result["methodology"],
    )

    return _build_generate_output(session.session_id, session.dashboard_version, "draft", result)


# ---------------------------------------------------------------------------
# POST /api/dashboard/confirm — Step 4: confirm dashboard & activate analyst
# ---------------------------------------------------------------------------

@router.post("/dashboard/confirm", response_model=DashboardConfirmOutput)
def dashboard_confirm(body: DashboardConfirmInput):
    session = _get_session_or_404(body.session_id)

    if session.dashboard_status not in ("draft", "edited_after_confirmation"):
        raise HTTPException(
            status_code=400,
            detail="Dashboard must be in draft state before confirming.",
        )

    analyst_context = build_analyst_context(
        dashboard_title=session.draft_spec.get("dashboard_title", "Dashboard") if session.draft_spec else "Dashboard",
        intent=session.draft_spec.get("intent", "") if session.draft_spec else "",
        kpi_cards=session.draft_kpi_cards,
        charts=session.draft_charts,
        insights=session.draft_insights,
        dataset_summaries=[],
        data_limitations=session.draft_spec.get("data_limitations", []) if session.draft_spec else [],
    )
    session.confirm(analyst_context)

    return DashboardConfirmOutput(
        session_id=session.session_id,
        dashboard_status="confirmed",
        dashboard_version=session.dashboard_version,
        analyst_chat_enabled=True,
    )


# ---------------------------------------------------------------------------
# POST /api/analyst-chat — Step 5: analyst follow-up
# ---------------------------------------------------------------------------

@router.post("/analyst-chat", response_model=AnalystChatOutput)
def analyst_chat(body: AnalystChatInput):
    session = _get_session_or_404(body.session_id)

    if not session.is_analyst_ready():
        raise HTTPException(
            status_code=400,
            detail="Please confirm the dashboard first before using the analyst chat.",
        )

    # Guardrails check
    allowed, refusal = check_analyst_question(body.user_question)
    if not allowed:
        return AnalystChatOutput(
            session_id=body.session_id,
            answer=refusal,
            scope_status="out_of_scope",
            dashboard_version_used=session.dashboard_version,
        )

    try:
        answer = answer_analyst_question(
            confirmed_context=session.confirmed_analyst_context,
            user_question=body.user_question,
        )
    except Exception as e:
        logger.error(f"analyst_chat error: {e}")
        raise HTTPException(status_code=500, detail="Analyst response failed.")

    return AnalystChatOutput(
        session_id=body.session_id,
        answer=answer,
        scope_status="in_scope",
        dashboard_version_used=session.dashboard_version,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_or_404(session_id: str):
    try:
        return require_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


def _build_generate_output(
    session_id: str,
    version: int,
    status_str: str,
    result: Dict[str, Any],
) -> DashboardGenerateOutput:
    return DashboardGenerateOutput(
        session_id=session_id,
        dashboard_id=session_id,
        dashboard_version=version,
        dashboard_status=status_str,
        dashboard_title=result["dashboard_title"],
        dashboard_title_ar=result.get("dashboard_title_ar"),
        intent=result["intent"],
        kpi_cards=result["kpi_cards"],
        charts=[
            {
                "id": c["id"],
                "title": c["title"],
                "type": c["type"],
                "figure": c.get("figure"),
                "error": c.get("error"),
                "reason": c["reason"],
                "summary": c.get("summary"),
            }
            for c in result["charts"]
        ],
        insights=result["insights"],
        methodology=result["methodology"],
        dataset_summaries=[
            {
                "dataset_id": d.get("dataset_id", ""),
                "name_en": d.get("name_en", ""),
                "name_ar": d.get("name_ar", ""),
                "record_count": d.get("record_count", 0),
                "date_range": d.get("date_range"),
                "key_stats": {k: str(v) for k, v in d.get("key_stats", {}).items()},
            }
            for d in result["dataset_summaries"]
        ],
        data_limitations=result["data_limitations"],
        analyst_chat_enabled=False,
    )
