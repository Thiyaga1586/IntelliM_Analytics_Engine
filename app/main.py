from fastapi import FastAPI, HTTPException, Query

from app.autonomous_engine import AutonomousEngine
from app.config import settings
from app.event_attributor import EventAttributor
from app.state_manager import StateManager
from app.utils import ensure_directories, read_csv_if_exists

app = FastAPI(title="Analytics Engine MVP", version="1.1.0")

engine = AutonomousEngine()
state_manager = engine.state_manager
event_attributor = EventAttributor()


@app.on_event("startup")
def startup_event() -> None:
    ensure_directories()
    StateManager()

    active_version = engine.model_manager.get_active_version()
    if active_version:
        state_manager.set_state("active_model_version", active_version)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "db_path": str(state_manager.db_path),
        "active_model_version": state_manager.get_state("active_model_version", ""),
        "last_ingested_date": state_manager.get_state("last_ingested_date", ""),
    }


@app.post("/ingest-actuals")
def ingest_actuals(date: str | None = Query(default=None)) -> dict:
    return engine.ingest_actuals(date=date)


@app.post("/forecast/refresh")
def refresh_forecast() -> dict:
    try:
        return engine.refresh_forecast()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast/latest")
def get_latest_forecast() -> dict:
    df = engine.forecast_manager.get_latest_forecast()
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/drift/status")
def get_drift_status() -> dict:
    return engine.drift_manager.latest_status() or {"status": "no_drift_data"}


@app.get("/models/active")
def get_active_model() -> dict:
    return engine.model_manager.get_active_model_info()


@app.post("/models/retrain")
def retrain_model() -> dict:
    return engine.trigger_retrain(trigger_reason="manual")


@app.post("/models/rollback")
def rollback_model() -> dict:
    result = engine.model_manager.rollback_to_last_stable()
    if not result.get("rolled_back"):
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return result


@app.get("/history/forecast-vs-actual")
def forecast_vs_actual_history(limit: int = Query(default=200, ge=1, le=5000)) -> dict:
    df = state_manager.get_forecast_vs_actual_history(limit=limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.post("/engine/run-cycle")
def run_cycle(date: str | None = Query(default=None)) -> dict:
    return engine.run_cycle(date=date)


@app.get("/api/filters")
def api_filters() -> dict:
    master_df = read_csv_if_exists(settings.MASTER_CSV_PATH)
    events_df = read_csv_if_exists(settings.EVENTS_CSV_PATH)

    entity_ids = sorted(master_df["entity_id"].astype(str).dropna().unique().tolist()) if "entity_id" in master_df.columns else []
    categories = sorted(master_df["category"].astype(str).dropna().unique().tolist()) if "category" in master_df.columns else []
    event_types = sorted(events_df["event_type"].astype(str).dropna().unique().tolist()) if "event_type" in events_df.columns else []
    dates = sorted(master_df["date"].astype(str).dropna().unique().tolist()) if "date" in master_df.columns else []

    return {
        "entity_ids": entity_ids,
        "categories": categories,
        "event_types": event_types,
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
    }


@app.get("/api/daily-summary")
def api_daily_summary(limit: int = Query(default=120, ge=1, le=5000)) -> dict:
    df = read_csv_if_exists(settings.DAILY_SUMMARY_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}

    if "date" in df.columns:
        df = df.sort_values("date")

    df = df.tail(limit).reset_index(drop=True)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/autonomous/status")
def api_autonomous_status() -> dict:
    active_model = engine.model_manager.get_active_model_info()
    latest_drift = engine.drift_manager.latest_status() or {}

    return {
        "status": "ok",
        "sim_day": int(state_manager.get_state("sim_day", "0")),
        "latest_date": state_manager.get_state("last_ingested_date", ""),
        "active_model_version": state_manager.get_state("active_model_version", ""),
        "rows_since_last_retrain": int(state_manager.get_state("rows_since_last_retrain", "0")),
        "drift_status": latest_drift.get("drift_status"),
        "mae": latest_drift.get("mae"),
        "mape": latest_drift.get("mape"),
        "rmse": latest_drift.get("rmse"),
        "model_info": active_model,
    }


@app.get("/api/realtime/drift")
def api_realtime_drift() -> dict:
    latest = engine.drift_manager.latest_status()
    if not latest:
        return {"status": "no_drift_data"}

    return {
        "status": "ok",
        "drift_ts": latest.get("drift_ts"),
        "model_version": latest.get("model_version"),
        "mae": latest.get("mae"),
        "mape": latest.get("mape"),
        "rmse": latest.get("rmse"),
        "max_feature_psi": latest.get("max_feature_psi"),
        "drift_status": latest.get("drift_status"),
    }


@app.get("/api/explain")
def api_explain(entity_id: str, target_date: str | None = Query(default=None)) -> dict:
    forecast_df = state_manager.get_latest_forecast(limit=5000)

    if forecast_df.empty:
        return {"status": "no_forecast_data", "entity_id": entity_id, "rows": []}

    forecast_df["entity_id"] = forecast_df["entity_id"].astype(str)
    out = forecast_df[forecast_df["entity_id"] == str(entity_id)].copy()

    if target_date:
        out = out[out["target_date"].astype(str) == str(target_date)]

    if out.empty:
        return {"status": "not_found", "entity_id": entity_id, "rows": []}

    explanation_rows = []
    for _, row in out.iterrows():
        likely_event = event_attributor.find_likely_event(
            entity_id=row["entity_id"],
            target_date=row["target_date"],
            signal_target="demand_index",
            lookback_days=5,
        )
        explanation_rows.append(
            {
                "entity_id": row["entity_id"],
                "target_date": row["target_date"],
                "pred_demand": row["pred_demand"],
                "pred_price": row["pred_price"],
                "model_version": row["model_version"],
                "explanation_text": row.get("explanation_text"),
                "shap_explanation": row.get("shap_explanation"),
                "likely_event": likely_event,
            }
        )

    return {
        "status": "ok",
        "entity_id": entity_id,
        "count": len(explanation_rows),
        "rows": explanation_rows,
    }


@app.get("/api/model-status")
def api_model_status() -> dict:
    return engine.model_manager.get_active_model_info()


@app.get("/api/runtime-state")
def api_runtime_state() -> dict:
    return state_manager.get_runtime_state_snapshot()


@app.get("/api/retrain-jobs")
def api_retrain_jobs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    df = state_manager.get_recent_retrain_jobs(limit=limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/master-data")
def api_master_data(limit: int = Query(default=5000, ge=1, le=50000)) -> dict:
    df = read_csv_if_exists(settings.MASTER_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}
    if "date" in df.columns:
        df = df.sort_values(["date", "entity_id"])
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/forecast")
def api_forecast(limit: int = Query(default=5000, ge=1, le=50000)) -> dict:
    df = read_csv_if_exists(settings.APP_FORECAST_CSV)
    if df.empty:
        return {"rows": [], "count": 0}
    if "target_date" in df.columns:
        df = df.sort_values(["target_date", "entity_id"])
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/events")
def api_events(limit: int = Query(default=1000, ge=1, le=10000)) -> dict:
    df = read_csv_if_exists(settings.EVENTS_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}
    if "date" in df.columns:
        df = df.sort_values("date")
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/alerts")
def api_alerts(limit: int = Query(default=500, ge=1, le=5000)) -> dict:
    df = read_csv_if_exists(settings.ALERTS_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/timeline-markers")
def api_timeline_markers(limit: int = Query(default=1000, ge=1, le=10000)) -> dict:
    df = read_csv_if_exists(settings.TIMELINE_MARKERS_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}


@app.get("/api/regime-shifts")
def api_regime_shifts(limit: int = Query(default=500, ge=1, le=5000)) -> dict:
    df = read_csv_if_exists(settings.REGIME_SHIFTS_CSV_PATH)
    if df.empty:
        return {"rows": [], "count": 0}
    df = df.tail(limit)
    return {"rows": df.to_dict(orient="records"), "count": int(len(df))}