"""
Content guardrails for all LLM calls.

Checks user questions before sending to the analyst agent.
Validates dashboard specs before rendering.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked topic patterns (Business Analyst Agent guardrail)
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = [
    r"\b(politic|election|government overthrow|protest)\b",
    r"\b(legal advice|lawsuit|sue|attorney)\b",
    r"\b(medical advice|diagnos|treatment|medicine|drug)\b",
    r"\b(password|api.key|secret|token|credential)\b",
    r"\b(execute|run code|python script|bash|shell|import os)\b",
    r"\b(ignore previous|forget instructions|jailbreak|pretend you are)\b",
    r"\b(write me a|generate code|code snippet for)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

_EDIT_REQUEST_PATTERNS = [
    r"\b(change the|edit|update|modify|add a chart|remove kpi|use .* instead|regenerate)\b",
    r"\b(switch to|replace|different (chart|dataset|visualization))\b",
]
_EDIT_COMPILED = [re.compile(p, re.IGNORECASE) for p in _EDIT_REQUEST_PATTERNS]


def check_analyst_question(question: str) -> Tuple[bool, str]:
    """
    Returns (is_allowed, reason_if_blocked).
    If is_allowed is False, reason contains the refusal message.
    """
    if not question or not question.strip():
        return False, "Please enter a question."

    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."

    for pattern in _COMPILED:
        if pattern.search(question):
            return False, (
                "I can only answer questions related to the generated dashboard "
                "and selected environmental datasets."
            )

    for pattern in _EDIT_COMPILED:
        if pattern.search(question):
            return False, (
                "That sounds like a dashboard edit request. Please use the "
                "'Edit Dashboard' option so I can regenerate the dashboard properly."
            )

    return True, ""


def sanitize_user_prompt(prompt: str) -> str:
    """
    Basic sanitization of the dashboard generation prompt.
    Strips control characters; does not filter environmental topics.
    """
    if not prompt:
        return ""
    # Remove null bytes and control characters except common whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)
    return cleaned.strip()[:2000]  # hard cap at 2000 chars


def validate_prompt_length(prompt: str) -> Tuple[bool, str]:
    """Check that the prompt is usable."""
    cleaned = sanitize_user_prompt(prompt)
    if len(cleaned) < 10:
        return False, "Your request is too short. Please describe what you want to see in more detail."
    if len(cleaned) > 2000:
        return False, "Your request is too long. Please shorten it to under 2000 characters."
    return True, ""
