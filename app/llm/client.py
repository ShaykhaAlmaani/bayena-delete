"""
LLM client with provider abstraction.

Currently wired to Groq. To switch providers, implement a new class
that follows the same interface and update get_llm_client().
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails or returns an unparseable response."""
    pass


class GroqClient:
    """
    Thin wrapper around the Groq Python SDK.
    Uses JSON mode to force structured output from the model.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Please copy .env.example to .env and add your key."
            )
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise LLMError("groq package is not installed. Run: pip install groq")
        self.model = model

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Call the model and force a JSON object response.
        Returns the parsed dict. Raises LLMError on failure.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content
            if not raw or not raw.strip():
                raise LLMError("LLM returned an empty response. Check your API key and model availability.")
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned invalid JSON: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")

    def chat_text(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Call the model and return a plain text response.
        Used for the Business Analyst Agent.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")


class MockLLMClient:
    """
    Fallback client that returns pre-defined mock responses.
    Used when GROQ_API_KEY is not set, so the UI still runs end-to-end.
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        logger.warning("Using MockLLMClient — set GROQ_API_KEY to enable real LLM calls.")

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        # Detect what kind of response is expected from the system prompt
        if "recommended_datasets" in system_prompt:
            return self._mock_interpret_result()
        return self._mock_dashboard_spec()

    def chat_text(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        return (
            "Direct answer: Based on the current dashboard, Riyadh shows the highest number of violations.\n\n"
            "Evidence: KPI 'Region with Highest Violations' displays Riyadh with 34% of total violations.\n\n"
            "Interpretation: Decision makers should prioritize enforcement and monitoring in the Riyadh region."
        )

    def _mock_interpret_result(self) -> Dict[str, Any]:
        return {
            "interpreted_intent": "Analyze environmental violations across regions and categories over time.",
            "recommended_datasets": [
                {
                    "dataset_id": "environmental_violations",
                    "dataset_name_en": "Environmental Violations Dataset",
                    "dataset_name_ar": "مجموعة بيانات المخالفات البيئية",
                    "relevance_reason": "Directly tracks violations by region, category, and date — matching the user's request.",
                    "confidence": 0.95,
                }
            ],
            "confidence_score": 0.9,
            "clarification_needed": False,
            "clarification_question": None,
        }

    def _mock_dashboard_spec(self) -> Dict[str, Any]:
        return {
            "dashboard_title": "Environmental Violations by Region and Category",
            "dashboard_title_ar": "المخالفات البيئية حسب المنطقة والفئة",
            "intent": "Analyze environmental violations across regions, categories, and time periods.",
            "selected_datasets": ["environmental_violations"],
            "kpis": [
                {"id": "kpi_total", "label": "Total Violations", "label_ar": "إجمالي المخالفات", "calculation": "count_rows", "column": "violation_id", "description": "Total number of recorded violations"},
                {"id": "kpi_top_region", "label": "Top Region", "label_ar": "أعلى منطقة", "calculation": "top_category_by_count", "column": "region", "description": "Region with most violations"},
                {"id": "kpi_top_category", "label": "Top Category", "label_ar": "أكثر فئة", "calculation": "top_category_by_count", "column": "category", "description": "Most common violation category"},
                {"id": "kpi_open", "label": "Open Violations", "label_ar": "مخالفات مفتوحة", "calculation": "count_rows", "column": "status", "description": "Violations still open"},
            ],
            "charts": [
                {"id": "chart_trend", "type": "line", "title": "Violations Over Time", "x": "date", "y": "count", "aggregation": "count", "color_by": None, "reason": "Shows the trend in violations over time."},
                {"id": "chart_region", "type": "bar", "title": "Violations by Region", "x": "region", "y": "count", "aggregation": "count", "color_by": None, "reason": "Compares violation counts across regions."},
            ],
            "insight_hints": [
                {"id": "ins_1", "type": "trend", "source_chart": "chart_trend", "description_hint": "Identify whether violations increased or decreased over the years."},
                {"id": "ins_2", "type": "comparison", "source_chart": "chart_region", "description_hint": "Note which region has the most violations and by how much."},
                {"id": "ins_3", "type": "anomaly", "source_chart": "chart_trend", "description_hint": "Look for any unusual spikes in violation counts."},
                {"id": "ins_4", "type": "recommendation", "source_chart": None, "description_hint": "Suggest policy action based on the top violation category."},
            ],
            "assumptions": ["Sample data used — not real government records."],
            "data_limitations": ["Date range limited to 2020–2024.", "Financial penalties may not reflect true fines."],
            "confidence_score": 0.92,
            "clarification_needed": False,
            "clarification_question": None,
        }


def get_llm_client(role: str = "dashboard") -> GroqClient | MockLLMClient:
    """
    Factory that returns the correct LLM client based on available config.

    role: "dashboard" → Dashboard Generation Agent model
          "analyst"   → Business Analyst Agent model
    """
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return MockLLMClient()

    model = (
        settings.DASHBOARD_AGENT_MODEL
        if role == "dashboard"
        else settings.ANALYST_AGENT_MODEL
    )
    return GroqClient(api_key=api_key, model=model)
