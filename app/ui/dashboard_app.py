"""
Bayena Dashboard UI — Dash application.

Multi-step wizard interface:
  Step 0: Landing — enter dashboard prompt
  Step 1: Confirm datasets recommended by the AI
  Step 2: View draft dashboard (KPIs, charts, insights) — edit or confirm
  Step 3: Confirmed dashboard + Business Analyst chat

Communicates with the FastAPI backend via HTTP (requests library).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Bayena color palette
# ---------------------------------------------------------------------------
NAVY = "#002B62"
DEEP_BLUE = "#013273"
DARK_BG = "#212944"
LIME = "#C5D75A"
MUTED_BLUE = "#365882"
OLIVE = "#989F97"
SAND = "#F1EEE7"
LIGHT_GRAY = "#E2E1DC"
WHITE = "#FCFCFC"

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Bayena | بيّنة",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose for gunicorn

# CSS is loaded automatically from app/ui/assets/custom.css by Dash

_CUSTOM_CSS_UNUSED = """
body {{ background-color: {SAND}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: {DARK_BG}; }}
.bayena-header {{ background: linear-gradient(135deg, {NAVY}, {DEEP_BLUE}); padding: 18px 32px; }}
.bayena-header h2 {{ color: {WHITE}; margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: 0.5px; }}
.bayena-header p {{ color: rgba(255,255,255,0.7); margin: 4px 0 0 0; font-size: 0.85rem; }}
.bayena-header .ar-title {{ color: {LIME}; font-size: 1.1rem; font-weight: 500; }}
.step-bar {{ background: {WHITE}; border-bottom: 2px solid {LIGHT_GRAY}; padding: 12px 32px; }}
.step-dot {{ width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; }}
.step-dot.active {{ background: {NAVY}; color: {WHITE}; }}
.step-dot.done {{ background: {LIME}; color: {DARK_BG}; }}
.step-dot.inactive {{ background: {LIGHT_GRAY}; color: {OLIVE}; }}
.step-label {{ font-size: 0.72rem; margin-top: 4px; }}
.step-label.active {{ color: {NAVY}; font-weight: 600; }}
.step-label.inactive {{ color: {OLIVE}; }}
.main-card {{ background: {WHITE}; border-radius: 12px; padding: 28px; box-shadow: 0 2px 12px rgba(0,43,98,0.07); margin-bottom: 20px; }}
.prompt-textarea {{ border: 2px solid {LIGHT_GRAY}; border-radius: 8px; padding: 14px; font-size: 1rem; width: 100%; resize: vertical; min-height: 110px; background: {WHITE}; color: {DARK_BG}; transition: border-color 0.2s; }}
.prompt-textarea:focus {{ border-color: {MUTED_BLUE}; outline: none; }}
.btn-primary-bayena {{ background: {NAVY}; color: {WHITE}; border: none; border-radius: 8px; padding: 10px 28px; font-weight: 600; font-size: 0.95rem; cursor: pointer; transition: background 0.2s; }}
.btn-primary-bayena:hover {{ background: {DEEP_BLUE}; }}
.btn-secondary-bayena {{ background: {WHITE}; color: {NAVY}; border: 2px solid {NAVY}; border-radius: 8px; padding: 8px 22px; font-weight: 500; font-size: 0.9rem; cursor: pointer; }}
.kpi-card {{ background: {WHITE}; border-radius: 10px; padding: 20px; border-left: 4px solid {LIME}; box-shadow: 0 2px 8px rgba(0,43,98,0.07); height: 100%; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
.kpi-label {{ font-size: 0.8rem; color: {OLIVE}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
.kpi-desc {{ font-size: 0.78rem; color: {OLIVE}; margin-top: 6px; }}
.dataset-badge {{ background: {SAND}; border: 1px solid {LIGHT_GRAY}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }}
.dataset-badge .ds-name {{ font-weight: 600; color: {NAVY}; font-size: 0.95rem; }}
.dataset-badge .ds-reason {{ font-size: 0.82rem; color: {OLIVE}; margin-top: 3px; }}
.confidence-pill {{ display: inline-block; background: {LIME}22; color: {DARK_BG}; border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }}
.insight-card {{ background: {SAND}; border-left: 3px solid {MUTED_BLUE}; border-radius: 4px 8px 8px 4px; padding: 14px 16px; margin-bottom: 10px; font-size: 0.88rem; color: {DARK_BG}; line-height: 1.55; }}
.insight-number {{ display: inline-block; width: 22px; height: 22px; background: {MUTED_BLUE}; color: white; border-radius: 50%; font-size: 0.72rem; font-weight: 700; text-align: center; line-height: 22px; margin-right: 8px; }}
.section-header {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: {OLIVE}; margin-bottom: 14px; }}
.methodology-box {{ background: #f8f7f4; border: 1px solid {LIGHT_GRAY}; border-radius: 8px; padding: 14px; font-size: 0.8rem; color: {OLIVE}; }}
.draft-badge {{ background: #FFF3CD; color: #856404; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }}
.confirmed-badge {{ background: #D1E7DD; color: #0A3622; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }}
.chat-container {{ background: {WHITE}; border-radius: 10px; border: 1px solid {LIGHT_GRAY}; overflow: hidden; }}
.chat-history {{ min-height: 200px; max-height: 380px; overflow-y: auto; padding: 16px; background: #f9f8f6; }}
.chat-history:empty {{ display: flex; align-items: center; justify-content: center; }}
.msg-user {{ background: {NAVY}; color: white; border-radius: 12px 12px 4px 12px; padding: 10px 14px; max-width: 75%; margin-left: auto; margin-bottom: 10px; font-size: 0.88rem; word-wrap: break-word; }}
.msg-bot {{ background: {WHITE}; color: {DARK_BG}; border: 1px solid {LIGHT_GRAY}; border-radius: 12px 12px 12px 4px; padding: 10px 14px; max-width: 85%; margin-right: auto; margin-bottom: 10px; font-size: 0.88rem; word-wrap: break-word; white-space: pre-line; }}
.chat-input-area {{ border-top: 1px solid {LIGHT_GRAY}; padding: 12px; background: {WHITE}; }}
.example-chip {{ display: inline-block; background: {WHITE}; border: 1px solid {LIGHT_GRAY}; border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; color: {MUTED_BLUE}; cursor: pointer; margin: 4px; transition: background 0.15s; }}
.example-chip:hover {{ background: {LIGHT_GRAY}; }}
.error-box {{ background: #FEE2E2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px 16px; color: #991B1B; font-size: 0.88rem; }}
.status-text {{ font-size: 0.8rem; color: {OLIVE}; }}
"""

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _header():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("بيّنة", className="ar-title"),
                html.H2("Bayena  Environmental Intelligence Platform"),
                html.P("AI-Powered Dashboard Generation  •  منصة البيانات البيئية الذكية"),
            ]),
            dbc.Col([
                html.Div("🌿 Saudi Environment Fund", style={"color": LIME, "fontSize": "0.8rem", "textAlign": "right", "marginTop": "12px"}),
            ], width=3),
        ], align="center"),
    ], className="bayena-header")


def _step_bar(active_step: int):
    steps = ["Enter Request", "Confirm Datasets", "Review Dashboard", "Analyst Chat"]
    cols = []
    for i, label in enumerate(steps):
        if i < active_step:
            dot_cls = "step-dot done"
            label_cls = "step-label active"
            icon = "✓"
        elif i == active_step:
            dot_cls = "step-dot active"
            label_cls = "step-label active"
            icon = str(i + 1)
        else:
            dot_cls = "step-dot inactive"
            label_cls = "step-label inactive"
            icon = str(i + 1)

        connector = html.Hr(style={"flex": 1, "borderColor": LIGHT_GRAY, "marginTop": "14px"}) if i < len(steps) - 1 else None
        cols.append(
            html.Div([
                html.Div([
                    html.Div(icon, className=dot_cls),
                    html.Div(label, className=label_cls, style={"textAlign": "center"}),
                ], style={"textAlign": "center"}),
            ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "minWidth": "90px"})
        )
        if connector:
            cols.append(html.Div(connector, style={"flex": 1, "padding": "0 4px", "marginTop": "-6px"}))

    return html.Div([
        html.Div(cols, style={"display": "flex", "alignItems": "flex-start", "justifyContent": "center", "padding": "16px 32px"}),
    ], className="step-bar")


def _example_prompts():
    examples = [
        "Air quality trends in Riyadh 2022–2024",
        "Environmental violations by region and category",
        "Water consumption by sector across regions",
        "Vegetation coverage changes over the last decade",
        "Waste recycling rates by region",
        "Climate indicators — temperature and rainfall trends",
    ]
    return html.Div([
        html.Div("Try an example:", style={"fontSize": "0.78rem", "color": OLIVE, "marginBottom": "6px"}),
        html.Div([
            html.Span(e, className="example-chip", id={"type": "example-chip", "index": i}, n_clicks=0)
            for i, e in enumerate(examples)
        ]),
    ], style={"marginTop": "12px"})


# ---------------------------------------------------------------------------
# Step layouts
# ---------------------------------------------------------------------------

def step_0_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("Generate an Environmental Dashboard", style={"color": NAVY, "fontWeight": 600}),
                    html.P(
                        "Describe what you want to understand. The AI will recommend the right datasets "
                        "and build an interactive dashboard from trusted environmental data.",
                        style={"color": OLIVE, "fontSize": "0.9rem"},
                    ),
                    html.Div([
                        dcc.Textarea(
                            id="prompt-input",
                            placeholder='e.g. "Show me environmental violations by region and category over time."',
                            className="prompt-textarea",
                        ),
                    ], style={"marginTop": "16px"}),
                    html.Div(id="prompt-error", style={"marginTop": "8px"}),
                    html.Div([
                        dbc.Button(
                            ["Analyze Request  →"],
                            id="btn-analyze",
                            color="primary",
                            style={"background": NAVY, "border": "none", "borderRadius": "8px",
                                   "padding": "10px 28px", "fontWeight": 600},
                            className="mt-3",
                        ),
                        html.Span(id="loading-indicator", style={"marginLeft": "12px", "color": OLIVE, "fontSize": "0.85rem"}),
                    ]),
                    _example_prompts(),
                ], className="main-card"),
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Div("Available Datasets", className="section-header"),
                    html.Div([
                        html.Div([
                            html.Div(f"• {d['dataset_name_en']}", style={"fontSize": "0.82rem", "color": DARK_BG, "marginBottom": "3px"}),
                        ])
                        for d in _get_catalog_preview()
                    ]),
                ], className="main-card"),
            ], md=4),
        ]),
    ])


def step_1_layout(session_data: Dict):
    datasets = session_data.get("recommended_datasets", [])
    intent = session_data.get("interpreted_intent", "")
    prompt = session_data.get("user_prompt", "")
    confidence = session_data.get("confidence_score", 0)

    dataset_blocks = []
    for ds in datasets:
        conf_pct = int(ds.get("confidence", 0) * 100)
        dataset_blocks.append(
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(ds.get("dataset_name_en", ""), className="ds-name"),
                        html.Div(ds.get("dataset_name_ar", ""), style={"fontSize": "0.82rem", "color": OLIVE, "direction": "rtl"}),
                        html.Div(ds.get("relevance_reason", ""), className="ds-reason"),
                    ]),
                    dbc.Col([
                        html.Span(f"{conf_pct}% match", className="confidence-pill"),
                    ], width=3, style={"textAlign": "right"}),
                ], align="center"),
            ], className="dataset-badge")
        )

    clarification = session_data.get("clarification_question")
    clarification_block = html.Div([
        html.Div("⚠ Clarification suggested:", style={"fontWeight": 600, "color": "#856404", "fontSize": "0.85rem"}),
        html.Div(clarification, style={"color": "#856404", "fontSize": "0.85rem"}),
    ], style={"background": "#FFF3CD", "borderRadius": "8px", "padding": "10px 14px", "marginBottom": "14px"}) if clarification else None

    return html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("Your Request", className="section-header"),
                    html.Div(f'"{prompt}"', style={"fontStyle": "italic", "color": DARK_BG, "marginBottom": "10px", "fontSize": "0.9rem"}),
                    html.Div("AI Interpretation", className="section-header", style={"marginTop": "12px"}),
                    html.Div(intent, style={"color": MUTED_BLUE, "fontSize": "0.88rem", "marginBottom": "16px"}),
                ]),
            ]),
            clarification_block,
            html.Div("Recommended Datasets", className="section-header"),
            html.Div(dataset_blocks),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Generate Dashboard  →",
                        id="btn-generate",
                        color="primary",
                        style={"background": NAVY, "border": "none", "borderRadius": "8px",
                               "padding": "10px 28px", "fontWeight": 600},
                        n_clicks=0,
                    ),
                ]),
                dbc.Col([
                    dbc.Button(
                        "← Edit Request",
                        id="btn-back-to-prompt",
                        outline=True,
                        color="secondary",
                        style={"borderRadius": "8px", "padding": "9px 22px"},
                        n_clicks=0,
                    ),
                ], width="auto"),
            ], style={"marginTop": "20px"}),
            html.Span(id="loading-generate", style={"marginLeft": "12px", "color": OLIVE, "fontSize": "0.85rem"}),
        ], className="main-card"),
    ])


def _kpi_cards_row(kpi_cards: List[Dict]):
    if not kpi_cards:
        return html.Div()
    cols = []
    for kpi in kpi_cards[:4]:
        cols.append(dbc.Col([
            html.Div([
                html.Div(kpi.get("label", ""), className="kpi-label"),
                html.Div(kpi.get("value", "—"), className="kpi-value"),
                html.Div(kpi.get("description", ""), className="kpi-desc"),
            ], className="kpi-card"),
        ], md=3, sm=6, style={"marginBottom": "16px"}))
    return dbc.Row(cols, className="g-3")


def _charts_row(charts: List[Dict]):
    chart_cols = []
    for chart in charts[:2]:
        figure = chart.get("figure")
        if figure:
            graph = dcc.Graph(
                figure=figure,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "340px"},
            )
        else:
            graph = html.Div([
                html.Div("⚠ Chart could not be rendered.", style={"color": OLIVE}),
                html.Div(chart.get("error", ""), style={"fontSize": "0.78rem", "color": OLIVE}),
            ], style={"height": "340px", "display": "flex", "flexDirection": "column",
                      "alignItems": "center", "justifyContent": "center", "background": SAND,
                      "borderRadius": "8px"})

        chart_cols.append(dbc.Col([
            html.Div([
                html.Div(chart.get("title", ""), style={"fontSize": "0.82rem", "color": OLIVE, "marginBottom": "6px"}),
                graph,
            ], style={"background": WHITE, "borderRadius": "10px", "padding": "16px",
                      "boxShadow": f"0 2px 8px rgba(0,43,98,0.07)"}),
        ], md=6, style={"marginBottom": "16px"}))

    return dbc.Row(chart_cols, className="g-3")


def _insights_block(insights: List[str]):
    if not insights:
        return html.Div()
    return html.Div([
        html.Div(className="section-header", children="Key Insights"),
        html.Div([
            html.Div([
                html.Span(str(i + 1), className="insight-number"),
                html.Span(insight),
            ], className="insight-card")
            for i, insight in enumerate(insights)
        ]),
    ])


def step_2_layout(session_data: Dict, is_confirmed: bool = False):
    title = session_data.get("dashboard_title", "Dashboard")
    title_ar = session_data.get("dashboard_title_ar", "")
    intent = session_data.get("intent", "")
    kpi_cards = session_data.get("kpi_cards", [])
    charts = session_data.get("charts", [])
    insights = session_data.get("insights", [])
    methodology = session_data.get("methodology", "")
    dataset_summaries = session_data.get("dataset_summaries", [])
    version = session_data.get("dashboard_version", 1)

    badge = html.Span("✓ Confirmed", className="confirmed-badge") if is_confirmed else html.Span("Draft", className="draft-badge")
    ds_labels = " • ".join(d.get("name_en", "") for d in dataset_summaries)

    action_row = html.Div([
        html.Div([
            dbc.Button(
                "✓ Confirm Dashboard",
                id="btn-confirm",
                color="success",
                style={"background": "#1A6B3C", "border": "none", "borderRadius": "8px",
                       "padding": "10px 28px", "fontWeight": 600, "marginRight": "12px"},
                n_clicks=0,
            ) if not is_confirmed else html.Div(),
            dbc.Button(
                "✏ Edit Dashboard",
                id="btn-edit-open",
                outline=True,
                color="secondary",
                style={"borderRadius": "8px", "padding": "9px 22px", "borderColor": MUTED_BLUE, "color": MUTED_BLUE},
                n_clicks=0,
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], style={"marginBottom": "16px"}) if not is_confirmed else html.Div()

    edit_area = html.Div([
        html.Div([
            dcc.Textarea(
                id="edit-prompt-input",
                placeholder='Describe the change, e.g. "Use region instead of category on the second chart."',
                style={"border": f"2px solid {LIGHT_GRAY}", "borderRadius": "8px", "padding": "10px",
                       "width": "100%", "minHeight": "80px", "resize": "vertical", "fontSize": "0.88rem"},
            ),
            dbc.Button(
                "Apply Edit  →",
                id="btn-apply-edit",
                color="primary",
                style={"background": MUTED_BLUE, "border": "none", "borderRadius": "8px",
                       "padding": "9px 22px", "fontWeight": 600, "marginTop": "8px"},
                n_clicks=0,
            ),
            html.Span(id="loading-edit", style={"marginLeft": "12px", "color": OLIVE, "fontSize": "0.85rem"}),
        ], className="main-card", style={"borderLeft": f"3px solid {MUTED_BLUE}"}),
    ], id="edit-area", style={"display": "none"})

    return html.Div([
        # Title & meta
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4([title, badge], style={"color": NAVY, "fontWeight": 600, "marginBottom": "4px"}),
                    html.Div(title_ar, style={"color": OLIVE, "fontSize": "0.9rem", "direction": "rtl", "marginBottom": "6px"}),
                    html.Div([
                        html.Span("v" + str(version), style={"fontSize": "0.75rem", "color": OLIVE, "marginRight": "12px"}),
                        html.Span(ds_labels, style={"fontSize": "0.75rem", "color": OLIVE}),
                    ]),
                    html.Div(intent, style={"fontSize": "0.82rem", "color": MUTED_BLUE, "marginTop": "8px"}),
                ]),
            ]),
        ], className="main-card"),

        action_row,
        edit_area,

        # KPIs
        html.Div([
            html.Div("Key Performance Indicators", className="section-header"),
            _kpi_cards_row(kpi_cards),
        ], className="main-card"),

        # Charts
        html.Div([
            html.Div("Visualizations", className="section-header"),
            _charts_row(charts),
        ], className="main-card"),

        # Insights
        html.Div([
            _insights_block(insights),
        ], className="main-card"),

        # Methodology
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div(methodology, className="methodology-box"),
                ], title="How was this dashboard generated?"),
            ], start_collapsed=True, flush=True),
        ], className="main-card", style={"padding": "12px 20px"}),
    ])


def step_3_chat_panel(session_id: str):
    return html.Div([
        html.Div("🤖 AI Analyst", className="section-header", style={"marginBottom": "8px", "fontSize": "0.7rem"}),
        html.P(
            "Ask questions about this dashboard. Answers are based on the confirmed data only.",
            style={"fontSize": "0.72rem", "color": OLIVE, "marginBottom": "10px", "lineHeight": "1.4"},
        ),
        html.Div([
            html.Div(id="chat-history-display", className="chat-history",
                     style={"minHeight": "300px", "maxHeight": "calc(100vh - 320px)"},
                     children=[html.Div("Ask a question about the dashboard...",
                                        style={"color": OLIVE, "fontSize": "0.75rem", "textAlign": "center", "padding": "20px"})]),
            html.Div([
                dcc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="e.g. Which region has the highest violations?",
                    style={"width": "100%", "border": f"1px solid {LIGHT_GRAY}",
                           "borderRadius": "6px", "padding": "7px 9px", "fontSize": "0.75rem",
                           "marginBottom": "7px"},
                    debounce=False,
                    n_submit=0,
                ),
                dbc.Button("Send", id="btn-send-chat", color="primary",
                           style={"background": NAVY, "border": "none", "borderRadius": "6px",
                                  "padding": "7px 0", "fontWeight": 600, "width": "100%",
                                  "fontSize": "0.78rem"},
                           n_clicks=0),
            ], className="chat-input-area"),
        ], className="chat-container"),
    ], className="main-card", style={"position": "sticky", "top": "16px"})


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

app.layout = html.Div([
    dcc.Store(id="store-step", data=0),
    dcc.Store(id="store-session", data={}),
    dcc.Store(id="store-dashboard", data={}),
    dcc.Store(id="store-chat-history", data=[]),

    _header(),
    html.Div(id="step-bar-container"),
    html.Div(id="main-content", style={"maxWidth": "1200px", "margin": "20px auto", "padding": "0 20px"}),
    html.Div([
        html.Div("Bayena / بيّنة  •  Saudi Environment Fund Hackathon  •  Sample Data — Not Real Government Records",
                 style={"textAlign": "center", "color": OLIVE, "fontSize": "0.75rem", "padding": "16px"}),
    ], style={"borderTop": f"1px solid {LIGHT_GRAY}", "marginTop": "40px"}),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Step bar update
@app.callback(Output("step-bar-container", "children"), Input("store-step", "data"))
def update_step_bar(step):
    return _step_bar(step)


# Main content renderer
@app.callback(
    Output("main-content", "children"),
    Input("store-step", "data"),
    State("store-session", "data"),
    State("store-dashboard", "data"),
)
def render_main(step, session_data, dashboard_data):
    if step == 0:
        return step_0_layout()
    elif step == 1:
        return step_1_layout(session_data)
    elif step == 2:
        return step_2_layout(dashboard_data, is_confirmed=False)
    elif step == 3:
        session_id = session_data.get("session_id", "")
        return html.Div([
            dbc.Row([
                dbc.Col(
                    step_2_layout(dashboard_data, is_confirmed=True),
                    width=7,
                ),
                dbc.Col(
                    step_3_chat_panel(session_id),
                    width=5,
                    style={"position": "sticky", "top": "0", "alignSelf": "flex-start"},
                ),
            ], className="g-2"),
        ])
    return html.Div("Invalid step.")


# Fill prompt from example chip
@app.callback(
    Output("prompt-input", "value"),
    [Input({"type": "example-chip", "index": i}, "n_clicks") for i in range(6)],
    prevent_initial_call=True,
)
def fill_example(*n_clicks_list):
    examples = [
        "Air quality trends in Riyadh 2022–2024",
        "Environmental violations by region and category",
        "Water consumption by sector across regions",
        "Vegetation coverage changes over the last decade",
        "Waste recycling rates by region",
        "Climate indicators — temperature and rainfall trends",
    ]
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"]
    try:
        idx = json.loads(triggered_id.split(".")[0])["index"]
        return examples[idx]
    except Exception:
        raise PreventUpdate


# Step 0 → 1: analyze request
@app.callback(
    Output("store-step", "data", allow_duplicate=True),
    Output("store-session", "data", allow_duplicate=True),
    Output("prompt-error", "children"),
    Output("loading-indicator", "children"),
    Input("btn-analyze", "n_clicks"),
    State("prompt-input", "value"),
    prevent_initial_call=True,
)
def analyze_request(n_clicks, prompt):
    if not n_clicks or not prompt or not prompt.strip():
        return dash.no_update, dash.no_update, html.Div("Please enter a description first.", className="error-box"), ""

    try:
        resp = requests.post(f"{BACKEND_URL}/api/dashboard/request", json={"user_prompt": prompt}, timeout=30)
        if resp.status_code != 200:
            err = resp.json().get("detail", "Request failed.")
            return dash.no_update, dash.no_update, html.Div(err, className="error-box"), ""
        data = resp.json()
        data["user_prompt"] = prompt
        return 1, data, "", ""
    except requests.exceptions.ConnectionError:
        return dash.no_update, dash.no_update, html.Div(
            "Cannot connect to the backend. Make sure the API server is running on port 8000.",
            className="error-box"
        ), ""
    except Exception as e:
        return dash.no_update, dash.no_update, html.Div(f"Error: {str(e)}", className="error-box"), ""


# Step 1 → 0: back to prompt
@app.callback(
    Output("store-step", "data", allow_duplicate=True),
    Input("btn-back-to-prompt", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_prompt(n):
    if not n:
        raise PreventUpdate
    return 0


# Step 1 → 2: generate dashboard
@app.callback(
    Output("store-step", "data", allow_duplicate=True),
    Output("store-dashboard", "data", allow_duplicate=True),
    Output("loading-generate", "children"),
    Input("btn-generate", "n_clicks"),
    State("store-session", "data"),
    prevent_initial_call=True,
)
def trigger_generate(n_clicks, session_data):
    if not n_clicks:
        raise PreventUpdate
    session_id = session_data.get("session_id", "")
    dataset_ids = [d["dataset_id"] for d in session_data.get("recommended_datasets", [])]
    if not session_id or not dataset_ids:
        return dash.no_update, dash.no_update, "Session error — please start again."

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/dashboard/generate",
            json={"session_id": session_id, "selected_dataset_ids": dataset_ids},
            timeout=60,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get('detail', 'Generation failed.')
            except Exception:
                detail = resp.text or f"Server error (HTTP {resp.status_code})"
            return dash.no_update, dash.no_update, html.Div(detail, className="error-box")
        return 2, resp.json(), ""
    except requests.exceptions.ConnectionError:
        return dash.no_update, dash.no_update, "Backend not reachable."
    except Exception as e:
        return dash.no_update, dash.no_update, f"Error: {str(e)}"


# Step 2: toggle edit area
@app.callback(
    Output("edit-area", "style"),
    Input("btn-edit-open", "n_clicks"),
    State("edit-area", "style"),
    prevent_initial_call=True,
)
def toggle_edit_area(n, current_style):
    if not n:
        raise PreventUpdate
    is_hidden = current_style.get("display") == "none"
    return {"display": "block"} if is_hidden else {"display": "none"}


# Step 2: apply edit
@app.callback(
    Output("store-dashboard", "data", allow_duplicate=True),
    Output("loading-edit", "children"),
    Input("btn-apply-edit", "n_clicks"),
    State("edit-prompt-input", "value"),
    State("store-session", "data"),
    prevent_initial_call=True,
)
def apply_edit(n_clicks, edit_text, session_data):
    if not n_clicks or not edit_text or not edit_text.strip():
        raise PreventUpdate
    session_id = session_data.get("session_id", "")
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/dashboard/edit",
            json={"session_id": session_id, "user_edit_request": edit_text},
            timeout=60,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get('detail', 'Failed.')
            except Exception:
                detail = resp.text or f"Server error (HTTP {resp.status_code})"
            return dash.no_update, f"Edit error: {detail}"
        return resp.json(), ""
    except Exception as e:
        return dash.no_update, f"Error: {str(e)}"


# Step 2 → 3: confirm dashboard
@app.callback(
    Output("store-step", "data", allow_duplicate=True),
    Output("store-dashboard", "data", allow_duplicate=True),
    Input("btn-confirm", "n_clicks"),
    State("store-session", "data"),
    State("store-dashboard", "data"),
    prevent_initial_call=True,
)
def confirm_dashboard(n_clicks, session_data, dashboard_data):
    if not n_clicks:
        raise PreventUpdate
    session_id = session_data.get("session_id", "")
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/dashboard/confirm",
            json={"session_id": session_id},
            timeout=15,
        )
        if resp.status_code != 200:
            raise PreventUpdate
        dashboard_data["analyst_chat_enabled"] = True
        return 3, dashboard_data
    except Exception:
        raise PreventUpdate


# Analyst chat: send message
@app.callback(
    Output("chat-history-display", "children"),
    Output("store-chat-history", "data"),
    Output("chat-input", "value"),
    Input("btn-send-chat", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("store-session", "data"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def send_chat(n_btn, n_sub, question, session_data, history):
    if not question or not question.strip():
        raise PreventUpdate
    session_id = session_data.get("session_id", "")
    history = history or []

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/analyst-chat",
            json={"session_id": session_id, "user_question": question},
            timeout=30,
        )
        if resp.status_code != 200:
            try:
                answer = f"Error: {resp.json().get('detail', 'Request failed.')}"
            except Exception:
                answer = f"Server error (HTTP {resp.status_code}): {resp.text[:200]}"
        else:
            answer = resp.json().get("answer", "No response.")
    except requests.exceptions.ConnectionError:
        answer = "Cannot connect to backend."
    except Exception as e:
        answer = f"Error: {str(e)}"

    history.append({"role": "user", "content": question})
    history.append({"role": "bot", "content": answer})

    chat_elements = []
    for msg in history:
        if msg["role"] == "user":
            chat_elements.append(html.Div(msg["content"], className="msg-user"))
        else:
            chat_elements.append(html.Div(msg["content"], className="msg-bot"))

    return chat_elements, history, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_catalog_preview():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/datasets", timeout=5)
        if resp.status_code == 200:
            return resp.json()[:7]
    except Exception:
        pass
    # Static fallback
    return [
        {"dataset_name_en": "Air Quality"},
        {"dataset_name_en": "Water Consumption"},
        {"dataset_name_en": "Environmental Violations"},
        {"dataset_name_en": "Vegetation Coverage"},
        {"dataset_name_en": "Waste Management"},
        {"dataset_name_en": "Protected Areas"},
        {"dataset_name_en": "Climate Indicators"},
    ]


# ---------------------------------------------------------------------------
# Entry point (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dash_port = int(os.getenv("DASH_PORT", "8050"))
    app.run(host="0.0.0.0", port=dash_port, debug=False)
