# Bayena / بيّنة — AI-Powered Environmental Dashboard Generation

> A smart environmental data platform for the **Saudi Environment Fund Hackathon**.
> Converts natural-language requests into interactive, data-grounded environmental dashboards using a two-agent AI architecture.

---

## What It Does

Bayena lets government analysts and environmental researchers generate rich dashboards by simply describing what they want in plain language:

> *"Show me environmental violations by region and category over time."*

The system:
1. Interprets the request using an AI agent
2. Recommends the most relevant environmental datasets from the catalog
3. Asks the user to confirm or adjust
4. Generates a full interactive dashboard — KPI cards, Plotly charts, written insights
5. Enables a scoped analyst chat for follow-up questions, grounded strictly in the generated dashboard

---

## Why It Matters

Environmental data in the region is often siloed, hard to access, and harder to interpret quickly. Bayena bridges the gap between raw datasets and decision-ready intelligence. Every insight is traceable to actual data — not LLM hallucination.

---

## Architecture: Two-Agent Design

The feature is built around two clearly separated AI agents:

### Agent 1 — Dashboard Generation Agent
- **Used during**: prompt interpretation, dataset recommendation, dashboard generation, dashboard edits
- **Returns**: A structured JSON dashboard specification (validated by Pydantic)
- **Never**: executes code, answers chat questions, or touches confirmed dashboards

### Agent 2 — Business Analyst Agent
- **Used during**: analyst chat (after dashboard confirmation only)
- **Receives**: the confirmed dashboard context (KPI values, chart summaries, insights, dataset stats)
- **Never**: modifies the dashboard, invents numbers, or answers unrelated questions

```
User Prompt
    │
    ▼
[Dashboard Generation Agent]  → JSON spec → Pydantic validation
    │
    ▼
Backend (chart_factory, kpi_calculator, insight_engine)
    │
    ▼
Draft Dashboard  →  User Confirms
    │
    ▼
[Business Analyst Agent]  ← confirmed context only
    │
    ▼
Analyst Chat Answers
```

The LLM **never executes code**. It only returns a structured JSON blueprint. The backend generates all actual calculations, charts, and insights using safe, predefined Python functions.

---

## Dashboard State Machine

```
awaiting_datasets  →  draft  →  confirmed
                         ↑           │
                         └── edited_after_confirmation
```

The analyst chat is only active when `dashboard_status == "confirmed"` and `analyst_context_version == dashboard_version`.

---

## User Flow

| Step | What Happens |
|------|-------------|
| 1    | User enters a natural-language request |
| 2    | System recommends datasets with confidence scores |
| 3    | User confirms or edits the recommendation |
| 4    | Draft dashboard generated: KPIs, 2 charts, 3–4 insights |
| 5    | User reviews — Edit or Confirm |
| 6    | Analyst chat activates on confirmation |
| 7    | User asks follow-up questions scoped to this dashboard |

---

## Project Structure

```
bayena-dashboard/
│
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Environment variable settings
│   │
│   ├── api/
│   │   ├── routes.py            # All FastAPI endpoints
│   │   └── schemas.py           # Request/response Pydantic models
│   │
│   ├── core/
│   │   ├── session_store.py     # In-memory session / state management
│   │   ├── dashboard_generator.py  # Pipeline orchestration
│   │   ├── kpi_calculator.py    # Safe KPI computation (no LLM code)
│   │   ├── chart_factory.py     # Safe Plotly chart builder
│   │   ├── insight_engine.py    # Data-driven written insights
│   │   └── guardrails.py        # Input validation & content filtering
│   │
│   ├── llm/
│   │   ├── client.py            # Groq API client + MockClient fallback
│   │   ├── prompts.py           # System prompts for both agents
│   │   └── structured_outputs.py  # Pydantic models for LLM JSON outputs
│   │
│   ├── data/
│   │   ├── catalog.py           # Dataset metadata + sample data generator
│   │   └── sample_data/         # Auto-generated CSVs (created on first run)
│   │
│   ├── ui/
│   │   └── dashboard_app.py     # Dash frontend (multi-step UI)
│   │
│   └── utils/
│       ├── logging.py
│       └── formatting.py
│
├── tests/
│   ├── test_guardrails.py
│   ├── test_dataset_selector.py
│   └── test_dashboard_generator.py
│
├── .env.example
├── requirements.txt
├── README.md
└── run.py
```

---

## API Endpoints

| Method | Endpoint | Agent | Description |
|--------|----------|-------|-------------|
| `GET`  | `/api/health` | — | Health check |
| `GET`  | `/api/datasets` | — | Full dataset catalog |
| `POST` | `/api/dashboard/request` | Generation | Interpret prompt, recommend datasets |
| `POST` | `/api/dashboard/generate` | Generation | Build draft dashboard |
| `POST` | `/api/dashboard/edit` | Generation | Edit draft dashboard |
| `POST` | `/api/dashboard/confirm` | — | Confirm dashboard, activate analyst |
| `POST` | `/api/analyst-chat` | Analyst | Follow-up Q&A (confirmed only) |

Full interactive docs at: `http://localhost:8000/docs`

---

## How to Run Locally

### 1. Clone / open the project

```bash
cd bayena-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# OR
source .venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
copy .env.example .env        # Windows
# OR
cp .env.example .env          # macOS / Linux
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_key_here
DASHBOARD_AGENT_MODEL=llama-3.3-70b-versatile
ANALYST_AGENT_MODEL=llama-3.1-8b-instant
```

> **Note:** If `GROQ_API_KEY` is not set, the system falls back to a `MockLLMClient` that returns pre-built demo responses. The full UI flow still works — great for offline demos.

### 5. Start both servers

```bash
python run.py
```

| Service | URL |
|---------|-----|
| Dashboard UI | http://localhost:8050 |
| API Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

### 6. Run tests

```bash
pytest tests/ -v
```

---

## How to Plug In a Real Groq API Key

1. Get your key from: https://console.groq.com/keys
2. Add it to `.env`:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```
3. Restart the app. The system automatically switches from MockClient to GroqClient.

The `get_llm_client()` factory in `app/llm/client.py` handles this switch transparently.

---

## How to Replace Mock Data with Real Datasets

The data catalog is in `app/data/catalog.py`. Each dataset has:
- Metadata (columns, tags, source, quality score)
- A generator function that creates sample CSV data

To replace with real data:
1. Add your CSV file to `app/data/sample_data/`
2. Update the `DATASET_CATALOG` entry to match your real column names
3. Remove or update the corresponding generator function

The rest of the pipeline (chart factory, KPI calculator, insight engine) reads column names from the spec — it adapts automatically as long as column names match the catalog.

---

## How Dashboard Generation Works

1. **User prompt** → `interpret_request()` → Dashboard Generation Agent (JSON mode) → `InterpretResult` (Pydantic-validated)
2. User confirms datasets → `generate_dashboard()`:
   - Dashboard Generation Agent returns `DashboardSpec` (JSON, Pydantic-validated)
   - Backend validates column existence in actual dataset
   - Auto-repairs broken column references
   - `chart_factory.build_chart()` creates Plotly figures using safe predefined functions
   - `kpi_calculator.compute_all_kpis()` computes values from real data
   - `insight_engine.generate_insights()` writes insights from actual statistics
3. User confirms → `build_analyst_context()` creates a text snapshot of the dashboard
4. Analyst Agent reads only from this snapshot — never the raw data or any unconfirmed spec

---

## LLM Safety Design

| Risk | How It's Mitigated |
|------|--------------------|
| LLM hallucinating data | LLM never touches real data. All numbers come from Pandas. |
| LLM executing arbitrary code | LLM only returns JSON specs. Backend executes using hardcoded functions only. |
| Stale analyst context | Analyst context is version-locked. Chat blocks if `analyst_context_version != dashboard_version`. |
| Prompt injection | Guardrails filter user questions before sending to analyst agent. |
| API key exposure | Keys loaded from env only. Never logged, never returned in API responses. |

---

## Dataset Catalog

| ID | Name | Key Columns |
|----|------|-------------|
| `air_quality` | Air Quality | pm25, pm10, aqi_value, city, region, date |
| `water_consumption` | Water Consumption | total_consumption_m3, per_capita_m3, region, sector |
| `vegetation_coverage` | Vegetation Coverage | coverage_pct, ndvi_index, region, year |
| `environmental_violations` | Environmental Violations | region, category, severity, status, date |
| `waste_management` | Waste Management | total_waste_tons, recycling_rate, region, date |
| `protected_areas` | Protected Areas | area_km2, type, region, year_established |
| `climate_indicators` | Climate Indicators | avg_temp_c, rainfall_mm, humidity_pct, dust_days |

All data is synthetically generated with fixed seed (reproducible). Values are realistic for Saudi Arabia.

---

## Future Improvements

- [ ] Replace in-memory session store with Redis
- [ ] Connect to real Saudi government environmental APIs
- [ ] Add map visualizations for geographic data (Plotly choropleth)
- [ ] Add Arabic-language UI toggle (RTL layout)
- [ ] Add export to PDF / Excel
- [ ] Add user authentication
- [ ] Add data quality scoring visualization
- [ ] Support multi-dataset joins for cross-indicator dashboards
- [ ] Add scheduled dashboard refresh
- [ ] Deploy to cloud (e.g., AWS App Runner / Azure Container Apps)

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Data | Pandas, NumPy |
| Visualization | Plotly, Dash, dash-bootstrap-components |
| LLM | Groq API (llama-3.3-70b / llama-3.1-8b-instant) |
| Validation | Pydantic v2 |
| Configuration | python-dotenv |
| Testing | pytest |

---

*Built for the Saudi Environment Fund Hackathon — Bayena / بيّنة Environmental Intelligence Platform*
