# IntelliM Analytics Engine

A production-oriented analytics backend for real-time market intelligence, forecasting, drift monitoring, and explainable decision support.

---

## Overview

The IntelliM Analytics Engine is a modular FastAPI-based analytics system designed to ingest market data, transform it into model-ready features, forecast future behavior, measure real-world drift against predictions, and expose structured APIs for dashboards and analyst workflows.

The engine addresses a common limitation in market intelligence systems: most solutions are either static reporting layers or isolated machine learning prototypes. This backend unifies both into a single operational pipeline that supports:

- historical analytics over curated market datasets
- incremental real-time style ingestion from a future simulation feed
- machine learning model lifecycle management
- forecast-versus-actual monitoring
- event-aware explainability
- dashboard-ready aggregation APIs

At its core, the project treats analytics as a continuously evolving system rather than a one-time model run. It combines data engineering, feature generation, model orchestration, state tracking, and serving logic into a cohesive backend suitable for live demos and extensible toward production deployment.

---

## Problem Statement

Modern product and market analytics workflows often suffer from four structural issues:

- data pipelines and predictive systems are disconnected
- predictions are not tracked against actual outcomes
- analytics dashboards are forced to reconstruct logic in the frontend
- business users receive model outputs without sufficient operational context

This backend solves those issues by introducing a stateful analytics engine that can:

- ingest the next available operational data batch
- compare predictions to actuals
- compute live drift and model health metrics
- retrain or rollback models when quality changes
- generate forecast outputs for downstream consumption
- provide explainability through both model-derived and event-derived narratives

---

## Objectives

The primary objectives of the backend are:

- build an end-to-end analytics service for structured market intelligence data
- support simulated real-time ingestion over historical and future datasets
- expose clean REST APIs for frontend dashboards
- maintain a persistent runtime state for forecasts, actuals, and model health
- support explainable outputs through event attribution and feature-based reasoning
- keep the architecture modular enough for future migration to cloud-native or database-heavy production setups

---

## Scope

The current scope includes:

- FastAPI application layer
- SQLite-backed runtime state
- CSV and JSON-based input/output layer
- feature engineering pipeline
- XGBoost-based demand modeling
- drift tracking and forecast-vs-actual comparison
- dashboard bundle aggregation endpoint
- deployment support through Docker and Hugging Face Spaces

The system is intentionally designed so that data storage, model storage, and analytics serving can later be upgraded independently.

---

## System Architecture

### High-Level Architecture

The backend is organized into five major layers:

1. **Data Input Layer** — reads historical market data, event metadata, summaries, and query simulation data from the `data/input` directory
2. **Feature & Modeling Layer** — transforms raw product-level observations into model-ready features; trains and serves demand prediction models
3. **Autonomous Execution Layer** — orchestrates ingestion, drift computation, retraining decisions, rollback logic, and forecast refresh
4. **State & Persistence Layer** — persists runtime state, actuals, predictions, forecast-vs-actual records, drift snapshots, and retrain jobs using SQLite
5. **API & Serving Layer** — exposes REST endpoints for raw data, forecasts, monitoring, explanations, and dashboard-ready aggregates

### Data Flow Summary

The engine follows a structured lifecycle:

```
Seed Data -> Feature Engineering -> Model Training -> Forecast Generation
         -> New Batch Ingestion -> Forecast vs Actual Comparison
         -> Drift Evaluation -> Retrain / Rollback Decision
         -> Updated Forecast + APIs for Frontend Consumption
```

### Runtime Interaction Model

At runtime, the backend behaves as an autonomous analytics loop:

- historical datasets are loaded as a baseline source
- a model is trained or loaded from the active registry
- a forecast is generated for the next target date
- the next batch from `query.csv` is ingested
- actual values are compared against prior forecasts
- drift metrics are computed and persisted
- retraining policies are evaluated
- the next forecast is regenerated
- outputs are made available through APIs and serving files

---

## Design Decisions and Rationale

### FastAPI for Service Layer

FastAPI provides strong developer ergonomics, clean typing support, high-performance request handling, and built-in OpenAPI documentation. This makes it ideal for a backend that serves both frontend integration and engineering validation.

### SQLite for Runtime State

SQLite is used as a lightweight operational state store for hackathon-grade deployment and local reproducibility. It keeps the engine self-contained while still enabling relational tracking of predictions, actuals, drift, and retrain history.

### CSV as Input/Serving Medium

CSV files are used for seed data, simulation input, and serving outputs because they are transparent, easy to inspect, demo-friendly, and sufficient for a prototype analytics platform. The architecture is structured so these can later be replaced by object storage, warehouses, or message-driven ingestion.

### Modular Engine Separation

Core concerns are split into dedicated modules:

- ingestion
- feature engineering
- forecasting
- drift tracking
- model management
- dashboard aggregation

This reduces coupling and keeps each subsystem independently testable and extensible.

---

## Tech Stack

### Languages

| Technology | Purpose |
|------------|---------|
| Python | Primary implementation language for analytics, orchestration, and APIs |

### Backend Frameworks

| Technology | Purpose | Why It Was Chosen |
|------------|---------|-------------------|
| FastAPI | HTTP API layer | Modern async-friendly framework with excellent typing and docs support |
| Uvicorn | ASGI server | Lightweight, fast, well-suited for FastAPI deployment |

### Data & Analytics Libraries

| Technology | Purpose | Why It Was Chosen |
|------------|---------|-------------------|
| pandas | Data loading, transformation, aggregation | Standard for structured analytics workflows |
| NumPy | Numeric operations | Efficient low-level numerical computation |
| scikit-learn | Metrics and ML utilities | Reliable model evaluation support |
| XGBoost | Demand forecasting model | Strong tabular performance and robust feature importance behavior |
| SHAP | Explainability support | Enables feature-level interpretation of model outputs |

### Persistence & Storage

| Technology | Purpose | Why It Was Chosen |
|------------|---------|-------------------|
| SQLite | Runtime state store | Lightweight persistent state without external DB setup |
| JSON | Model metadata and registry format | Human-readable and easy to version |
| CSV | Seed and serving datasets | Transparent and frontend-friendly |

### Deployment & Tooling

| Technology | Purpose | Why It Was Chosen |
|------------|---------|-------------------|
| Docker | Containerized deployment | Consistent deployment environment |
| Hugging Face Spaces | Live hosting | Fast public deployment for demos |
| Git LFS | Large file handling | Required for large seed datasets in Git-backed repos |

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── autonomous_engine.py
│   ├── clear_prediction.py
│   ├── config.py
│   ├── dashboard_bundle.py
│   ├── drift_manager.py
│   ├── event_attributor.py
│   ├── feature_builder.py
│   ├── forecast_manager.py
│   ├── main.py
│   ├── model_manager.py
│   ├── realtime_ingestor.py
│   ├── state_manager.py
│   └── utils.py
├── data/
│   ├── input/
│   │   ├── app_alerts.csv
│   │   ├── app_daily_summary.csv
│   │   ├── app_explanations.csv
│   │   ├── app_master_clean.csv
│   │   ├── app_regime_shifts.csv
│   │   ├── app_timeline_markers.csv
│   │   ├── events.csv
│   │   ├── feature_metadata.json
│   │   └── query.csv
│   ├── models/
│   │   └── xgb_demand_*/
│   ├── output/
│   │   ├── app_forecast.csv
│   │   ├── forecast_vs_actual.csv
│   │   └── realtime_drift_status.csv
│   └── model_registry.json
├── scripts/
│   └── init_db.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Module Breakdown

### `app/main.py`

Primary FastAPI entrypoint. Registers routes for health, ingestion, forecasting, drift, explanations, model operations, data serving, and the bundled dashboard endpoint.

### `app/autonomous_engine.py`

Central orchestration module. Coordinates:

- ingestion
- master/daily summary updates
- forecast-vs-actual logging
- drift computation
- alert generation
- retrain and rollback decisions
- forecast refresh

### `app/realtime_ingestor.py`

Consumes the next eligible date batch from `query.csv`, validates daily completeness, updates runtime state, and ensures the simulation advances strictly forward.

### `app/feature_builder.py`

Transforms raw historical and live batch data into model-ready features. Responsible for:

- feature contract loading
- lag and rolling features
- regime and event-aware enrichment
- training and live inference frames

### `app/model_manager.py`

Owns model lifecycle operations:

- active model lookup
- artifact loading
- model training and validation
- model versioning
- acceptance/rejection logic
- rollback to last stable model

### `app/forecast_manager.py`

Generates the next forecast for each entity using the active model. Produces:

- demand predictions
- carried-forward price context
- explanation text
- optional SHAP-based explanations

### `app/drift_manager.py`

Computes forecast-vs-actual comparisons and model health metrics such as:

- MAE
- RMSE
- MAPE
- feature PSI
- drift status

### `app/state_manager.py`

Encapsulates SQLite persistence and schema initialization. Stores:

- runtime state
- actuals
- predictions
- forecast-vs-actual records
- drift snapshots
- retrain jobs

### `app/event_attributor.py`

Finds likely events associated with changes in observed or forecasted behavior by scanning event windows and selecting the best candidate explanation.

### `app/dashboard_bundle.py`

Builds a single aggregated dashboard payload containing:

- KPIs
- weekly series
- category breakdowns
- next-day comparisons
- scatter and radar views
- competitor views
- event summaries
- regime summaries
- report metadata

### `app/utils.py`

Shared utilities for:

- atomic writes
- CSV and JSON loading
- seed bootstrap
- directory initialization
- input normalization

### `scripts/init_db.py`

Bootstraps required directories, registry files, and SQLite schema. Used during local setup and container startup.

---

## Core Features

### 1. Historical Data Serving

The backend serves structured historical datasets for:

- product-level trend analysis
- event overlays
- regime shifts
- dashboard summaries
- alert feeds

### 2. Real-Time Style Batch Ingestion

The system simulates real-time behavior by ingesting the next date batch from `query.csv`. Each batch:

- contains all products for a date
- is inserted into runtime state
- updates serving data
- triggers drift comparison
- triggers forecast refresh

### 3. Forecasting Engine

Demand forecasting is handled by XGBoost with a model registry and versioned artifacts. The engine supports:

- retraining from historical + ingested actuals
- forecast generation for the next target date
- persistence of forecast rows and metadata

### 4. Drift Monitoring

After actuals arrive, the system computes:

- demand error
- price error
- absolute error metrics
- drift snapshots
- feature PSI

This makes model quality observable over time.

### 5. Explainability

The backend supports two complementary explanation paths:

**Feature-Based Explanation**

Forecast outputs may include SHAP-derived reasoning for the most influential predictors.

**Event-Based Attribution**

Relevant events near a target date are linked back to the forecasted or observed behavior to create business-readable narratives.

### 6. Model Registry and Rollback

Every trained model is stored as a versioned artifact with:

- metadata
- feature columns
- baseline stats
- serialized model file

The registry tracks:

- active version
- last stable version
- historical versions

Rollback is supported when drift or quality policies indicate regression risk.

### 7. Frontend Convenience APIs

In addition to modular endpoints, the backend exposes a bundled dashboard endpoint so the frontend can render complex views with a single request.

---

## API Design

### API Characteristics

- REST-style routes
- JSON responses
- frontend-friendly payload shapes
- dashboard bundle support for reduced client orchestration
- operational endpoints for model management and engine execution

### Core Endpoints

#### Health and Status

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Root status and endpoint summary |
| GET | `/health` | Service health and active runtime information |
| GET | `/api/autonomous/status` | Current simulation day, latest ingested date, model and drift summary |
| GET | `/api/realtime/drift` | Latest drift metrics |
| GET | `/api/runtime-state` | Full runtime state snapshot |
| GET | `/api/model-status` | Active model metadata |

#### Engine Operations

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/engine/run-cycle` | Execute one full ingest-drift-forecast cycle |
| POST | `/ingest-actuals` | Ingest a specified or next batch |
| POST | `/forecast/refresh` | Refresh forecasts from latest available context |
| POST | `/models/retrain` | Train and activate a candidate model |
| POST | `/models/rollback` | Roll back to the last stable model |

#### Raw and Derived Data APIs

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/filters` | Return filterable entities, categories, date ranges, and event types |
| GET | `/api/master-data` | Historical product-level records |
| GET | `/api/daily-summary` | Daily market overview rows |
| GET | `/api/events` | Event timeline data |
| GET | `/api/alerts` | Alert rows for live feed |
| GET | `/api/timeline-markers` | Marker overlays for charts |
| GET | `/api/regime-shifts` | Regime shift annotations |
| GET | `/api/forecast` | Latest forecast rows |
| GET | `/history/forecast-vs-actual` | Forecast-vs-actual history |
| GET | `/api/explain` | Explanation output for an entity and optional date |
| GET | `/api/dashboard-bundle` | Aggregated dashboard payload |

### Example Requests

**Run One Analytics Cycle**

```
POST /engine/run-cycle
```

**Explain a Forecasted Entity**

```
GET /api/explain?entity_id=101&target_date=2025-04-02
```

**Fetch Dashboard Bundle**

```
GET /api/dashboard-bundle?category=all&weeks=12&days=7
```

### Request Validation

Validation is currently performed through:

- query parameter constraints in FastAPI
- schema expectations in data loaders
- completeness checks in the ingestion engine
- defensive numeric coercion in analytics modules

### Authentication

The current prototype does not include user authentication or role-based authorization. The API is intended for controlled demo or internal environments. Security extensions are described in the future improvements section.

---

## Data Flow and Processing Logic

### Seed Data Sources

The system begins with curated inputs in `data/input`:

- `app_master_clean.csv` — historical master dataset
- `query.csv` — future simulation feed
- `events.csv` — event metadata for attribution
- `app_daily_summary.csv` — daily summary seed
- `app_alerts.csv` — existing alert feed seed
- `app_regime_shifts.csv` — regime shift metadata
- `app_timeline_markers.csv` — chart marker seed
- `feature_metadata.json` — feature contract for modeling

### Detailed Processing Flow

#### 1. Initialization

At startup:

- directories are created
- registry file is bootstrapped if missing
- SQLite schema is initialized
- runtime state keys are seeded

#### 2. Historical Data Load

Historical master data is loaded and normalized:

- date coercion
- numeric coercion
- entity normalization
- duplicate elimination where required

#### 3. Feature Construction

The feature builder creates:

- lagged demand features
- rolling means and volatility
- event flags
- change-point and regime-aware features
- interaction terms
- calendar encodings

#### 4. Model Training

On retrain:

- historical and actual data are merged
- features are built
- train/validation split is performed by date
- XGBoost is trained and evaluated
- artifacts are saved under a versioned directory
- active model registry is updated if accepted

#### 5. Forecast Generation

The forecast manager:

- resolves the active model
- loads latest context
- builds live inference features
- predicts next-day demand
- writes `app_forecast.csv`
- persists predictions in SQLite

#### 6. Ingestion

The ingestion engine:

- reads `query.csv`
- finds the next unconsumed date
- validates batch completeness
- inserts actuals
- advances `sim_day` and `last_ingested_date`

#### 7. Forecast-vs-Actual Comparison

The drift manager:

- aligns prior predictions with actual batch rows
- computes demand and price errors
- stores forecast-vs-actual records
- exports `forecast_vs_actual.csv`

#### 8. Drift Snapshot

The latest drift snapshot is computed using:

- MAE
- RMSE
- MAPE
- PSI against stored training baselines

#### 9. Retrain / Rollback Decision

The autonomous engine evaluates model policy:

- **stable**: no action
- **degrading**: retrain
- **rollback condition met**: revert to last stable model

#### 10. Dashboard Serving

The API layer serves:

- modular raw datasets
- explanation endpoints
- bundled dashboard aggregates

---

## Setup and Installation

### Prerequisites

- Python 3.10 or 3.11 recommended
- Git
- Git LFS for large CSV files
- optional: Docker for containerized deployment

### Local Installation

#### 1. Clone the Repository

```bash
git clone <repo-url>
cd IntelliM_Analytics_Engine
```

#### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Initialize Directories and Database

```bash
python -m scripts.init_db
```

#### 5. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Open API Documentation

```
http://127.0.0.1:8000/docs
```

### Docker-Based Run

```bash
docker build -t intellim-analytics-engine .
docker run -p 7860:7860 intellim-analytics-engine
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANALYTICS_DATA_DIR` | Runtime data directory | `./data` in local usage, `/tmp/data` in containerized demo mode |
| `API_HOST` | Bind host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

---

## Scalability and Performance Considerations

### Current Strategy

The current implementation prioritizes:

- deterministic local reproducibility
- transparent file-based analytics
- low operational complexity
- fast deployability for demos

### Scalability Characteristics

**Strengths**

- modular service decomposition
- model artifact versioning
- API/engine separation
- runtime state abstraction through `StateManager`
- clear migration path from file-based to service-backed storage

**Known Constraints**

- SQLite is suitable for single-instance deployments but not horizontal write-heavy scaling
- CSV-backed historical sources are not ideal for very large datasets
- retraining is synchronous and not yet offloaded to a background worker
- SHAP can be expensive for larger feature matrices

### Planned Production Upgrades

Future production scaling can include:

- PostgreSQL for runtime state
- object storage for model artifacts and large serving files
- Redis or Kafka for event-driven ingestion
- background workers for retraining and scheduled jobs
- feature store integration
- partitioned analytical storage for historical datasets

---

## Error Handling and Logging

### Current Reliability Strategy

The backend uses:

- defensive file existence checks
- numeric coercion and fallback defaults
- idempotent upserts in SQLite
- batch completeness validation for ingestion
- graceful empty-data returns for dashboard and explanation endpoints

### Initialization Resilience

The database bootstrap script wraps initialization in `try/except` so deployment failures surface clearly.

### Runtime Resilience

Examples of guarded behavior include:

- returning empty frames instead of hard-failing on missing serving files
- falling back from actual-based forecast context to historical context
- preserving model registry integrity during model selection
- rejecting incomplete query batches before state mutation

### Observability Recommendations

For fuller production readiness, the next layer should include:

- structured logging with request correlation IDs
- explicit error classes
- metrics export for model and engine operations
- tracing for ingestion and forecast pipelines

---

## Security Considerations

### Current State

This project is designed for controlled environments and demo deployments. It currently does not implement:

- user authentication
- token-based API access
- role-based authorization
- request throttling

### Data Safety Considerations

- runtime data is isolated in a configurable data directory
- model artifacts are versioned and auditable
- SQLite state avoids direct in-memory-only mutation
- file writes use atomic write patterns for critical outputs

### Recommended Security Enhancements

For production deployment, the following should be added:

- API authentication via JWT or API keys
- route-level authorization
- HTTPS-only deployment
- secrets managed via environment variables or secret stores
- request validation hardening
- audit logging for model operations
- rate limiting on expensive endpoints
- input sanitation for external file ingestion

---

## Future Improvements

### Platform and Infrastructure

- migrate runtime state from SQLite to PostgreSQL
- introduce object storage for model and output artifacts
- support scheduled/background jobs for forecast refresh and retraining
- add CI/CD validation for data contracts and API smoke tests

### Modeling

- add explicit multi-horizon forecasting
- integrate sequence models such as LSTM or temporal transformers
- implement challenger-champion evaluation policies
- add category-specific or ensemble model routing

### Explainability

- improve causal event attribution scoring
- connect model explanations with regime and event windows
- generate richer narrative summaries for analyst workflows

### Data Engineering

- formalize schemas with pydantic or pandera validation
- move from CSV seed files to structured data ingestion contracts
- add automated data quality checks and anomaly detection upstream

### Frontend Support

- expose typed response schemas for bundle endpoints
- add pagination and query filters to large endpoints
- support websocket or polling strategies for live updates

### Operations

- structured logging and metrics
- health probes and readiness checks
- model and drift dashboards for operators
- reproducible deployment profiles for Spaces, Railway, and container platforms

---

## Conclusion

IntelliM Analytics Engine is more than a data-serving API. It is a structured analytics runtime that combines ingestion, feature engineering, predictive modeling, state management, drift monitoring, explainability, and dashboard-oriented aggregation into a unified backend.

Its design demonstrates production-grade engineering instincts even within a hackathon context:

- modular architecture
- explicit runtime state
- versioned model management
- explainability-aware outputs
- deployment-ready service boundaries

The system is intentionally pragmatic: simple enough to run and demo quickly, but architected with enough separation of concerns to evolve into a more scalable production platform. It provides a strong foundation for market intelligence applications that require not only data access, but operational prediction, monitoring, and decision support.
