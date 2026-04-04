import os
from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.getenv("ANALYTICS_DATA_DIR", str(BASE_DIR / "data")))

    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    MODELS_DIR = DATA_DIR / "models"

    DB_PATH = DATA_DIR / "runtime.db"
    REGISTRY_PATH = DATA_DIR / "model_registry.json"
    FEATURE_METADATA_PATH = INPUT_DIR / "feature_metadata.json"

    REQUIRED_INPUT_COLUMNS = [
        "date",
        "entity_id",
    ]

    OPTIONAL_INPUT_COLUMNS = [
        "event_type",
        "event_effect_hint",
    ]

    QUERY_CSV_PATH = INPUT_DIR / "query.csv"
    MASTER_CSV_PATH = INPUT_DIR / "app_master_clean.csv"
    EVENTS_CSV_PATH = INPUT_DIR / "events.csv"
    REGIME_SHIFTS_CSV_PATH = INPUT_DIR / "app_regime_shifts.csv"
    DAILY_SUMMARY_CSV_PATH = INPUT_DIR / "app_daily_summary.csv"
    ALERTS_CSV_PATH = INPUT_DIR / "app_alerts.csv"
    TIMELINE_MARKERS_CSV_PATH = INPUT_DIR / "app_timeline_markers.csv"
    EXPLANATIONS_CSV_PATH = INPUT_DIR / "app_explanations.csv"

    # frontend-serving outputs
    APP_FORECAST_CSV = OUTPUT_DIR / "app_forecast.csv"
    FORECAST_VS_ACTUAL_CSV = OUTPUT_DIR / "forecast_vs_actual.csv"
    REALTIME_DRIFT_CSV = OUTPUT_DIR / "realtime_drift_status.csv"

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    TARGET_COLUMN = "demand_index"

    MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "150"))
    ERROR_WINDOW = int(os.getenv("ERROR_WINDOW", "200"))
    FORECAST_EXPORT_LIMIT = int(os.getenv("FORECAST_EXPORT_LIMIT", "500"))
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "500"))

    RETRAIN_MAE_MULTIPLIER = float(os.getenv("RETRAIN_MAE_MULTIPLIER", "1.25"))
    ROLLBACK_MAE_MULTIPLIER = float(os.getenv("ROLLBACK_MAE_MULTIPLIER", "1.15"))
    RETRAIN_MAPE_THRESHOLD = float(os.getenv("RETRAIN_MAPE_THRESHOLD", "0.20"))
    FEATURE_PSI_THRESHOLD = float(os.getenv("FEATURE_PSI_THRESHOLD", "0.20"))
    FEATURE_PSI_SEVERE_THRESHOLD = float(os.getenv("FEATURE_PSI_SEVERE_THRESHOLD", "0.30"))

    RETRAIN_MIN_NEW_ROWS = int(os.getenv("RETRAIN_MIN_NEW_ROWS", "100"))
    RETRAIN_SCHEDULE_DAYS = int(os.getenv("RETRAIN_SCHEDULE_DAYS", "14"))
    RETRAIN_SCHEDULE_MIN_ROWS = int(os.getenv("RETRAIN_SCHEDULE_MIN_ROWS", "150"))
    RETRAIN_COOLDOWN_DAYS = int(os.getenv("RETRAIN_COOLDOWN_DAYS", "3"))


settings = Settings()