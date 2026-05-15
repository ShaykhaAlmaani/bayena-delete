"""
Chart factory: safely builds Plotly figures from validated ChartSpec objects.

No arbitrary code is executed. All chart types and aggregations are
hardcoded to predefined Plotly functions. The LLM only supplies
a validated specification — it never executes code.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.llm.structured_outputs import ChartSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bayena color palette for charts
# ---------------------------------------------------------------------------

BAYENA_COLORS = [
    "#002B62", "#365882", "#C5D75A", "#013273",
    "#989F97", "#4A7DAB", "#7BA3C5", "#D4E0B0",
]

CHART_TEMPLATE = dict(
    paper_bgcolor="#FCFCFC",
    plot_bgcolor="#F8F7F4",
    font=dict(family="Segoe UI, Arial, sans-serif", color="#212944", size=12),
    colorway=BAYENA_COLORS,
    title=dict(font=dict(color="#002B62", size=15), x=0.02),
    xaxis=dict(gridcolor="#E2E1DC", linecolor="#C8C7C3", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#E2E1DC", linecolor="#C8C7C3", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(**CHART_TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# Core aggregation helper
# ---------------------------------------------------------------------------

def _aggregate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregation: str,
    color_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate the dataframe for chart rendering.
    Returns a simple dataframe with x, y (and optional color) columns.
    """
    group_cols = [x_col]
    if color_col and color_col in df.columns and color_col != x_col:
        group_cols.append(color_col)

    if aggregation == "count":
        agg_df = df.groupby(group_cols).size().reset_index(name="count")
        result_y = "count"
    else:
        if y_col not in df.columns:
            # Fall back to any numeric column
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError(f"No numeric columns available for aggregation in dataset.")
            y_col = numeric_cols[0]
            logger.warning(f"Requested column '{y_col}' not found — using '{numeric_cols[0]}'.")

        agg_func = {"sum": "sum", "mean": "mean", "max": "max", "min": "min"}.get(aggregation, "sum")
        agg_df = df.groupby(group_cols)[y_col].agg(agg_func).reset_index()
        result_y = y_col

    return agg_df, result_y


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _build_line_chart(spec: ChartSpec, df: pd.DataFrame) -> go.Figure:
    x_col = spec.x
    if x_col in df.columns:
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col]).sort_values(x_col)
        # Resample to monthly for cleaner lines
        df[x_col] = df[x_col].dt.to_period("M").dt.to_timestamp()

    try:
        agg_df, y_col = _aggregate(df, x_col, spec.y, spec.aggregation, spec.color_by)
    except Exception as e:
        raise ValueError(f"Line chart aggregation failed: {e}")

    color_col = spec.color_by if spec.color_by and spec.color_by in agg_df.columns else None
    fig = px.line(
        agg_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=spec.title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        color_discrete_sequence=BAYENA_COLORS,
    )
    fig.update_traces(line=dict(width=2.5), mode="lines+markers", marker=dict(size=4))
    return _apply_template(fig)


def _build_bar_chart(spec: ChartSpec, df: pd.DataFrame) -> go.Figure:
    x_col = spec.x
    try:
        agg_df, y_col = _aggregate(df, x_col, spec.y, spec.aggregation, spec.color_by)
    except Exception as e:
        raise ValueError(f"Bar chart aggregation failed: {e}")

    agg_df = agg_df.sort_values(y_col, ascending=False)
    color_col = spec.color_by if spec.color_by and spec.color_by in agg_df.columns else None
    fig = px.bar(
        agg_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=spec.title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        color_discrete_sequence=BAYENA_COLORS,
    )
    fig.update_traces(marker_line_width=0)
    return _apply_template(fig)


def _build_donut_chart(spec: ChartSpec, df: pd.DataFrame) -> go.Figure:
    cat_col = spec.x
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found in dataset.")

    counts = df[cat_col].value_counts().reset_index()
    counts.columns = [cat_col, "count"]
    counts = counts.head(8)  # limit slices

    fig = go.Figure(go.Pie(
        labels=counts[cat_col],
        values=counts["count"],
        hole=0.45,
        marker=dict(colors=BAYENA_COLORS, line=dict(color="#FCFCFC", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=spec.title, font=dict(color="#002B62", size=15), x=0.02),
        **{k: v for k, v in CHART_TEMPLATE.items() if k not in ["xaxis", "yaxis"]},
    )
    return fig


def _build_scatter_chart(spec: ChartSpec, df: pd.DataFrame) -> go.Figure:
    x_col = spec.x
    y_col = spec.y
    if x_col not in df.columns or y_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("Not enough numeric columns for scatter chart.")
        x_col, y_col = numeric_cols[0], numeric_cols[1]

    color_col = spec.color_by if spec.color_by and spec.color_by in df.columns else None
    fig = px.scatter(
        df.dropna(subset=[x_col, y_col]),
        x=x_col,
        y=y_col,
        color=color_col,
        title=spec.title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        color_discrete_sequence=BAYENA_COLORS,
        opacity=0.7,
    )
    return _apply_template(fig)


def _build_heatmap(spec: ChartSpec, df: pd.DataFrame) -> go.Figure:
    x_col = spec.x
    y_col = spec.y if spec.y in df.columns else df.select_dtypes(include="number").columns[0]
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Heatmap requires valid x and y columns.")

    pivot = df.pivot_table(values=y_col, index=spec.color_by or "region", columns=x_col, aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#E2E1DC"], [0.5, "#365882"], [1, "#002B62"]],
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text=spec.title, font=dict(color="#002B62", size=15), x=0.02),
        **{k: v for k, v in CHART_TEMPLATE.items() if k not in ["xaxis", "yaxis"]},
    )
    return fig


# ---------------------------------------------------------------------------
# Public build function
# ---------------------------------------------------------------------------

CHART_BUILDERS = {
    "line": _build_line_chart,
    "bar": _build_bar_chart,
    "donut": _build_donut_chart,
    "scatter": _build_scatter_chart,
    "heatmap": _build_heatmap,
}


def build_chart(spec: ChartSpec, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a Plotly figure from a validated ChartSpec.
    Returns the figure as a dict (compatible with dcc.Graph).
    Raises ValueError on invalid spec or data issues.
    """
    builder = CHART_BUILDERS.get(spec.type)
    if not builder:
        raise ValueError(f"Unsupported chart type: {spec.type}")

    try:
        import json as _json
        fig = builder(spec, df)
        # Use to_json() then parse to strip numpy types, ensuring JSON-safe output
        return _json.loads(fig.to_json())
    except Exception as e:
        logger.error(f"Chart build failed for spec '{spec.id}': {e}")
        raise ValueError(f"Could not build chart '{spec.title}': {e}")


def build_chart_summary(spec: ChartSpec, df: pd.DataFrame) -> str:
    """
    Returns a human-readable summary of what a chart shows.
    Used to populate the analyst context.
    """
    try:
        x_col = spec.x
        if spec.aggregation == "count":
            if x_col in df.columns:
                counts = df.groupby(x_col).size().sort_values(ascending=False)
                top_key = counts.index[0]
                top_val = counts.iloc[0]
                return (
                    f"{spec.type.title()} chart of {spec.aggregation} by {x_col}. "
                    f"Highest value: {top_key} ({top_val:,}). "
                    f"Total data points: {len(counts)}."
                )
        elif spec.y in df.columns:
            val = df[spec.y].agg(spec.aggregation)
            return (
                f"{spec.type.title()} chart of {spec.aggregation}({spec.y}) by {x_col}. "
                f"Overall {spec.aggregation}: {val:,.1f}."
            )
    except Exception:
        pass
    return f"{spec.type.title()} chart — {spec.title}."
