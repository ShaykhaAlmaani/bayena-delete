"""
In-memory session store for dashboard sessions.

Each session tracks the full lifecycle of a dashboard:
draft → confirmed → edited_after_confirmation → confirmed again.

For production, replace with Redis or a database-backed store.
"""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DashboardSession:
    """
    Represents one active dashboard generation session.

    dashboard_status:
      - "awaiting_datasets"      : user prompted, datasets recommended
      - "draft"                  : dashboard generated, not yet confirmed
      - "confirmed"              : user confirmed the dashboard
      - "edited_after_confirmation": user re-opened for edit after confirming
    """

    def __init__(self, session_id: str, user_prompt: str):
        self.session_id: str = session_id
        self.user_prompt: str = user_prompt
        self.dashboard_status: str = "awaiting_datasets"
        self.dashboard_version: int = 0

        # Set during interpret step
        self.interpreted_intent: Optional[str] = None
        self.recommended_datasets: List[Dict[str, Any]] = []

        # Set during generate/edit steps
        self.selected_dataset_ids: List[str] = []
        self.draft_spec: Optional[Dict[str, Any]] = None
        self.draft_kpi_cards: List[Dict[str, Any]] = []
        self.draft_charts: List[Dict[str, Any]] = []      # Plotly figure dicts
        self.draft_insights: List[str] = []
        self.draft_methodology: Optional[str] = None

        # Set on confirmation — this is what the analyst reads from
        self.confirmed_spec: Optional[Dict[str, Any]] = None
        self.confirmed_kpi_cards: List[Dict[str, Any]] = []
        self.confirmed_charts: List[Dict[str, Any]] = []
        self.confirmed_insights: List[str] = []
        self.confirmed_analyst_context: Optional[str] = None
        self.analyst_context_version: int = -1           # must match dashboard_version

        self.created_at: str = datetime.utcnow().isoformat()
        self.updated_at: str = datetime.utcnow().isoformat()

    def _touch(self):
        self.updated_at = datetime.utcnow().isoformat()

    def set_interpreted(
        self,
        intent: str,
        recommended_datasets: List[Dict[str, Any]],
    ) -> None:
        self.interpreted_intent = intent
        self.recommended_datasets = recommended_datasets
        self.dashboard_status = "awaiting_datasets"
        self._touch()

    def set_draft(
        self,
        selected_dataset_ids: List[str],
        spec: Dict[str, Any],
        kpi_cards: List[Dict[str, Any]],
        charts: List[Dict[str, Any]],
        insights: List[str],
        methodology: str,
    ) -> None:
        self.selected_dataset_ids = selected_dataset_ids
        self.draft_spec = spec
        self.draft_kpi_cards = kpi_cards
        self.draft_charts = charts
        self.draft_insights = insights
        self.draft_methodology = methodology
        self.dashboard_version += 1
        self.dashboard_status = "draft"

        # Invalidate any previous confirmation
        self.confirmed_spec = None
        self.confirmed_analyst_context = None
        self.analyst_context_version = -1
        self._touch()

    def confirm(self, analyst_context: str) -> None:
        self.confirmed_spec = self.draft_spec
        self.confirmed_kpi_cards = self.draft_kpi_cards
        self.confirmed_charts = self.draft_charts
        self.confirmed_insights = self.draft_insights
        self.confirmed_analyst_context = analyst_context
        self.analyst_context_version = self.dashboard_version
        self.dashboard_status = "confirmed"
        self._touch()

    def begin_edit(self) -> None:
        if self.dashboard_status == "confirmed":
            self.dashboard_status = "edited_after_confirmation"
        self.confirmed_spec = None
        self.confirmed_analyst_context = None
        self.analyst_context_version = -1
        self._touch()

    def is_analyst_ready(self) -> bool:
        return (
            self.dashboard_status == "confirmed"
            and self.analyst_context_version == self.dashboard_version
            and self.confirmed_analyst_context is not None
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_prompt": self.user_prompt,
            "dashboard_status": self.dashboard_status,
            "dashboard_version": self.dashboard_version,
            "selected_datasets": self.selected_dataset_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "analyst_ready": self.is_analyst_ready(),
        }


# ---------------------------------------------------------------------------
# Session store (in-memory)
# ---------------------------------------------------------------------------

_active_sessions: Dict[str, DashboardSession] = {}


def create_session(user_prompt: str) -> DashboardSession:
    session_id = str(uuid.uuid4())
    session = DashboardSession(session_id=session_id, user_prompt=user_prompt)
    _active_sessions[session_id] = session
    logger.info(f"Created session: {session_id}")
    return session


def get_session(session_id: str) -> Optional[DashboardSession]:
    return _active_sessions.get(session_id)


def require_session(session_id: str) -> DashboardSession:
    session = _active_sessions.get(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    return session


def delete_session(session_id: str) -> None:
    _active_sessions.pop(session_id, None)


def active_session_count() -> int:
    return len(_active_sessions)
