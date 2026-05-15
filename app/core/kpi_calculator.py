"""
KPI calculator: safely computes KPI card values from validated KPIPlan objects.

All calculation types are predefined and hardcoded.
The LLM only selects which calculation to use — it never executes code.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from app.llm.structured_outputs import KPIPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allowed calculation functions
# ---------------------------------------------------------------------------

def _count_rows(df: pd.DataFrame, column: str) -> Any:
    return len(df)


def _sum(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return df[column].sum()


def _mean(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return round(df[column].mean(), 2)


def _max(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return df[column].max()


def _min(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return df[column].min()


def _top_category_by_count(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    counts = df[column].value_counts()
    if counts.empty:
        return "N/A"
    top = counts.index[0]
    pct = round(counts.iloc[0] / len(df) * 100, 1)
    return f"{top} ({pct}%)"


def _count_unique(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return df[column].nunique()


def _latest_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return "N/A"
    return df[column].iloc[-1] if not df.empty else "N/A"


def _pct_change(df: pd.DataFrame, column: str) -> Any:
    """Year-over-year change — counts by year."""
    try:
        df = df.copy()
        date_col = "date" if "date" in df.columns else ("year" if "year" in df.columns else None)
        if not date_col:
            return "N/A"
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df = df.dropna(subset=[date_col])
        df["year"] = df[date_col].dt.year
        yearly = df.groupby("year").size()
        if len(yearly) < 2:
            return "N/A"
        last = yearly.iloc[-1]
        prev = yearly.iloc[-2]
        if prev == 0:
            return "N/A"
        change = round((last - prev) / prev * 100, 1)
        sign = "+" if change >= 0 else ""
        return f"{sign}{change}%"
    except Exception as e:
        logger.warning(f"pct_change calculation failed: {e}")
        return "N/A"


_CALCULATION_MAP = {
    "count_rows": _count_rows,
    "sum": _sum,
    "mean": _mean,
    "max": _max,
    "min": _min,
    "top_category_by_count": _top_category_by_count,
    "count_unique": _count_unique,
    "latest_value": _latest_value,
    "pct_change": _pct_change,
}


# ---------------------------------------------------------------------------
# KPI card builder
# ---------------------------------------------------------------------------

def compute_kpi(plan: KPIPlan, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a single KPI card value from a validated plan.
    Returns a dict ready for the API response and UI rendering.
    """
    calc_fn = _CALCULATION_MAP.get(plan.calculation)
    if not calc_fn:
        logger.warning(f"Unknown calculation type: {plan.calculation}")
        return {
            "id": plan.id,
            "label": plan.label,
            "label_ar": plan.label_ar or plan.label,
            "value": "N/A",
            "description": plan.description or "",
            "calculation": plan.calculation,
        }

    try:
        value = calc_fn(df, plan.column)
    except Exception as e:
        logger.error(f"KPI calculation error for '{plan.id}': {e}")
        value = "Error"

    if isinstance(value, float):
        value = f"{value:,.2f}"
    elif isinstance(value, int):
        value = f"{value:,}"

    return {
        "id": plan.id,
        "label": plan.label,
        "label_ar": plan.label_ar or plan.label,
        "value": str(value),
        "description": plan.description or "",
        "calculation": plan.calculation,
    }


def compute_all_kpis(plans: List[KPIPlan], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute all KPI cards for the dashboard."""
    return [compute_kpi(plan, df) for plan in plans]
