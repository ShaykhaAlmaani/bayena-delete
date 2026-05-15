"""
Insight engine: generates written business-analyst-style insights
from data, chart specs, and KPI values.

Insights are computed programmatically using Pandas statistics.
The LLM only provides hints about what to look for (InsightHint).
The actual numbers always come from the data.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from app.llm.structured_outputs import InsightHint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual insight generators by type
# ---------------------------------------------------------------------------

def _trend_insight(df: pd.DataFrame, hint: InsightHint, chart_specs: List[Dict]) -> str:
    try:
        date_col = next((c for c in ["date", "year", "month"] if c in df.columns), None)
        if not date_col:
            return hint.description_hint

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df["year"] = df[date_col].dt.year
        yearly = df.groupby("year").size()
        if len(yearly) < 2:
            return "Limited time data available for trend analysis."

        first_year, last_year = yearly.index[0], yearly.index[-1]
        first_val, last_val = yearly.iloc[0], yearly.iloc[-1]
        direction = "increased" if last_val > first_val else "decreased"
        pct = abs(round((last_val - first_val) / first_val * 100, 1)) if first_val > 0 else 0
        peak_year = yearly.idxmax()
        peak_val = int(yearly.max())

        return (
            f"Records {direction} by {pct}% from {first_year} ({first_val:,}) to "
            f"{last_year} ({last_val:,}). Peak year was {peak_year} with {peak_val:,} records. "
            f"This {'upward' if last_val > first_val else 'downward'} trend warrants continued monitoring."
        )
    except Exception as e:
        logger.warning(f"Trend insight failed: {e}")
        return hint.description_hint


def _comparison_insight(df: pd.DataFrame, hint: InsightHint, chart_specs: List[Dict]) -> str:
    try:
        cat_col = None
        for chart in chart_specs:
            if chart.get("id") == hint.source_chart:
                cat_col = chart.get("x")
                break
        if not cat_col or cat_col not in df.columns:
            cat_col = next((c for c in ["region", "category", "sector", "type"] if c in df.columns), None)
        if not cat_col:
            return hint.description_hint

        counts = df[cat_col].value_counts()
        top_key = counts.index[0]
        top_val = int(counts.iloc[0])
        total = int(counts.sum())
        top_pct = round(top_val / total * 100, 1)

        insight = f"{top_key} leads with {top_val:,} records ({top_pct}% of total)."
        if len(counts) > 1:
            second_key = counts.index[1]
            second_val = int(counts.iloc[1])
            insight += f" {second_key} follows with {second_val:,} ({round(second_val/total*100,1)}%)."
        if len(counts) >= 3:
            lowest_key = counts.index[-1]
            lowest_val = int(counts.iloc[-1])
            insight += f" {lowest_key} has the fewest with {lowest_val:,} records."
        return insight
    except Exception as e:
        logger.warning(f"Comparison insight failed: {e}")
        return hint.description_hint


def _anomaly_insight(df: pd.DataFrame, hint: InsightHint, chart_specs: List[Dict]) -> str:
    try:
        date_col = next((c for c in ["date", "year"] if c in df.columns), None)
        if not date_col:
            return hint.description_hint

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df = df.dropna(subset=[date_col])
        df["month_year"] = df[date_col].dt.to_period("M")
        monthly = df.groupby("month_year").size()
        if monthly.empty:
            return hint.description_hint

        mean_val = monthly.mean()
        std_val = monthly.std()
        peak_period = str(monthly.idxmax())
        peak_val = int(monthly.max())

        if std_val > 0 and (peak_val - mean_val) / std_val > 1.5:
            z = round((peak_val - mean_val) / std_val, 1)
            return (
                f"A notable spike occurred in {peak_period} with {peak_val:,} records — "
                f"{z} standard deviations above the monthly average of {mean_val:.0f}. "
                "This may indicate a targeted enforcement campaign, seasonal pattern, or data anomaly."
            )
        return (
            f"The busiest period was {peak_period} with {peak_val:,} records, compared to "
            f"a monthly average of {mean_val:.0f}."
        )
    except Exception as e:
        logger.warning(f"Anomaly insight failed: {e}")
        return hint.description_hint


def _recommendation_insight(df: pd.DataFrame, hint: InsightHint, kpi_cards: List[Dict]) -> str:
    try:
        region_kpi = next(
            (k for k in kpi_cards if "region" in k.get("id", "").lower() or "region" in k.get("label", "").lower()), None
        )
        cat_kpi = next(
            (k for k in kpi_cards if "categor" in k.get("id", "").lower() or "type" in k.get("label", "").lower()), None
        )
        parts = []
        if region_kpi:
            parts.append(f"Priority attention should be directed to {region_kpi['value']}, which shows the highest activity.")
        if cat_kpi:
            parts.append(f"The most prevalent category is {cat_kpi['value']}, suggesting targeted policy measures are needed.")
        if not parts:
            parts.append("Decision makers should focus on the highest-volume segments identified in this dashboard.")
        parts.append("Regular monitoring and clear KPI targets are recommended to track improvement over time.")
        return " ".join(parts)
    except Exception as e:
        logger.warning(f"Recommendation insight failed: {e}")
        return hint.description_hint


def _summary_insight(df: pd.DataFrame, hint: InsightHint, kpi_cards: List[Dict]) -> str:
    try:
        count_kpi = next((k for k in kpi_cards if k.get("calculation") == "count_rows"), None)
        total = count_kpi["value"] if count_kpi else f"{len(df):,}"
        date_col = next((c for c in ["date", "year"] if c in df.columns), None)
        date_range_str = ""
        if date_col:
            df_c = df.copy()
            df_c[date_col] = pd.to_datetime(df_c[date_col].astype(str), errors="coerce")
            years = df_c[date_col].dt.year.dropna()
            if not years.empty:
                date_range_str = f" covering {int(years.min())}–{int(years.max())}"
        return (
            f"This dashboard analyzes {total} records{date_range_str}. "
            "The data provides a comprehensive view across multiple regions and categories. "
            "Key indicators are summarized in the KPI cards above."
        )
    except Exception as e:
        logger.warning(f"Summary insight failed: {e}")
        return hint.description_hint


_INSIGHT_GENERATORS = {
    "trend": _trend_insight,
    "comparison": _comparison_insight,
    "anomaly": _anomaly_insight,
    "recommendation": _recommendation_insight,
    "summary": _summary_insight,
}


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def generate_insights(
    hints: List[InsightHint],
    df: pd.DataFrame,
    chart_specs: List[Dict[str, Any]],
    kpi_cards: List[Dict[str, Any]],
) -> List[str]:
    """
    Generate written insights from InsightHints and actual data.
    Returns a list of insight strings ready for display.
    """
    insights = []
    for hint in hints:
        generator = _INSIGHT_GENERATORS.get(hint.type)
        if not generator:
            insights.append(hint.description_hint)
            continue
        try:
            if hint.type in ("recommendation", "summary"):
                text = generator(df, hint, kpi_cards)
            else:
                text = generator(df, hint, chart_specs)
            insights.append(text)
        except Exception as e:
            logger.error(f"Failed to generate insight '{hint.id}': {e}")
            insights.append(hint.description_hint)
    return insights
