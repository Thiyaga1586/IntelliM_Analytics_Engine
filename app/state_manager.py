import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.utils import ensure_directories, utc_now_iso


class StateManager:
    def __init__(self, db_path: Path | None = None):
        ensure_directories()
        self.db_path = str(db_path or settings.DB_PATH)
        self._ensure_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_predictions_log_columns(self, conn: sqlite3.Connection) -> None:
        existing = pd.read_sql_query("PRAGMA table_info(predictions_log)", conn)
        existing_cols = set(existing["name"].tolist())

        required_additions = {
            "event_type": "TEXT",
            "event_effect_hint": "TEXT",
            "explanation_text": "TEXT",
            "shap_explanation": "TEXT",
        }

        for col, col_type in required_additions.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {col_type}")

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_target_entity_model
            ON predictions_log(target_date, entity_id, model_version)
            """
        )

    def _ensure_forecast_vs_actual_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_fva_date_entity_model
            ON forecast_vs_actual(date, entity_id, model_version)
            """
        )

    def _ensure_tables(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS actuals_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    demand_index REAL,
                    price_index REAL,
                    sentiment_index REAL,
                    search_index REAL,
                    ad_index REAL,
                    event_type TEXT,
                    event_effect_hint TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(date, entity_id)
                );

                CREATE TABLE IF NOT EXISTS predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_ts TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    pred_demand REAL NOT NULL,
                    pred_price REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    event_type TEXT,
                    event_effect_hint TEXT,
                    explanation_text TEXT,
                    shap_explanation TEXT
                );

                CREATE TABLE IF NOT EXISTS forecast_vs_actual (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    pred_demand REAL,
                    actual_demand REAL,
                    demand_error REAL,
                    abs_demand_error REAL,
                    pred_price REAL,
                    actual_price REAL,
                    price_error REAL,
                    abs_price_error REAL,
                    event_type TEXT,
                    event_effect_hint TEXT,
                    model_version TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS drift_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drift_ts TEXT NOT NULL,
                    model_version TEXT,
                    mae REAL,
                    mape REAL,
                    rmse REAL,
                    max_feature_psi REAL,
                    drift_status TEXT,
                    payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS retrain_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_reason TEXT,
                    old_version TEXT,
                    new_version TEXT,
                    status TEXT,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_actuals_date ON actuals_log(date);
                CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions_log(target_date);
                CREATE INDEX IF NOT EXISTS idx_fva_date ON forecast_vs_actual(date);
                CREATE INDEX IF NOT EXISTS idx_fva_entity ON forecast_vs_actual(entity_id);
                CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_summary(drift_ts);
                CREATE INDEX IF NOT EXISTS idx_retrain_created_at ON retrain_jobs(created_at);
                """
            )

            self._ensure_predictions_log_columns(conn)
            self._ensure_forecast_vs_actual_indexes(conn)
            conn.commit()

        self._bootstrap_state()

    def _bootstrap_state(self) -> None:
        defaults = {
            "sim_day": "0",
            "last_retrain_sim_day": "0",
            "rows_since_last_retrain": "0",
            "consecutive_perf_drift": "0",
            "consecutive_rollback_drift": "0",
            "active_baseline_mae": "999999",
            "active_model_version": "",
            "last_ingested_date": "",
        }
        for key, value in defaults.items():
            if self.get_state(key) is None:
                self.set_state(key, value)

    def read_sql(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_value FROM runtime_state WHERE state_key = ?",
                (key,),
            ).fetchone()
        return row["state_value"] if row else default

    def set_state(self, key: str, value: Any) -> None:
        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (state_key, state_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at = excluded.updated_at
                """,
                (key, str(value), now),
            )
            conn.commit()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def insert_actuals(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        rows = []
        now = utc_now_iso()

        for _, row in df.iterrows():
            rows.append(
                (
                    str(row["date"]),
                    str(row["entity_id"]),
                    self._safe_float(row.get("demand_index")),
                    self._safe_float(row.get("price_index")),
                    self._safe_float(row.get("sentiment_index")),
                    self._safe_float(row.get("search_index")),
                    self._safe_float(row.get("ad_index")),
                    row.get("event_type"),
                    row.get("event_effect_hint"),
                    now,
                )
            )

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO actuals_log
                (date, entity_id, demand_index, price_index, sentiment_index,
                 search_index, ad_index, event_type, event_effect_hint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, entity_id) DO UPDATE SET
                    demand_index = excluded.demand_index,
                    price_index = excluded.price_index,
                    sentiment_index = excluded.sentiment_index,
                    search_index = excluded.search_index,
                    ad_index = excluded.ad_index,
                    event_type = excluded.event_type,
                    event_effect_hint = excluded.event_effect_hint
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def insert_predictions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        payload = df.copy()

        required_cols = [
            "prediction_ts",
            "target_date",
            "entity_id",
            "pred_demand",
            "pred_price",
            "model_version",
            "event_type",
            "event_effect_hint",
            "explanation_text",
            "shap_explanation",
        ]

        for col in required_cols:
            if col not in payload.columns:
                payload[col] = None

        rows = []
        for _, row in payload.iterrows():
            rows.append(
                (
                    str(row["prediction_ts"]),
                    str(row["target_date"]),
                    str(row["entity_id"]),
                    self._safe_float(row.get("pred_demand")),
                    self._safe_float(row.get("pred_price")),
                    str(row["model_version"]),
                    row.get("event_type"),
                    row.get("event_effect_hint"),
                    row.get("explanation_text"),
                    row.get("shap_explanation"),
                )
            )

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO predictions_log
                (
                    prediction_ts,
                    target_date,
                    entity_id,
                    pred_demand,
                    pred_price,
                    model_version,
                    event_type,
                    event_effect_hint,
                    explanation_text,
                    shap_explanation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date, entity_id, model_version) DO UPDATE SET
                    prediction_ts = excluded.prediction_ts,
                    pred_demand = excluded.pred_demand,
                    pred_price = excluded.pred_price,
                    event_type = excluded.event_type,
                    event_effect_hint = excluded.event_effect_hint,
                    explanation_text = excluded.explanation_text,
                    shap_explanation = excluded.shap_explanation
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def insert_forecast_vs_actual(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        rows = []
        now = utc_now_iso()

        for _, row in df.iterrows():
            rows.append(
                (
                    str(row["date"]),
                    str(row["entity_id"]),
                    self._safe_float(row.get("pred_demand")),
                    self._safe_float(row.get("actual_demand")),
                    self._safe_float(row.get("demand_error")),
                    self._safe_float(row.get("abs_demand_error")),
                    self._safe_float(row.get("pred_price")),
                    self._safe_float(row.get("actual_price")),
                    self._safe_float(row.get("price_error")),
                    self._safe_float(row.get("abs_price_error")),
                    row.get("event_type"),
                    row.get("event_effect_hint"),
                    row.get("model_version"),
                    now,
                )
            )

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO forecast_vs_actual
                (
                    date,
                    entity_id,
                    pred_demand,
                    actual_demand,
                    demand_error,
                    abs_demand_error,
                    pred_price,
                    actual_price,
                    price_error,
                    abs_price_error,
                    event_type,
                    event_effect_hint,
                    model_version,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, entity_id, model_version) DO UPDATE SET
                    pred_demand = excluded.pred_demand,
                    actual_demand = excluded.actual_demand,
                    demand_error = excluded.demand_error,
                    abs_demand_error = excluded.abs_demand_error,
                    pred_price = excluded.pred_price,
                    actual_price = excluded.actual_price,
                    price_error = excluded.price_error,
                    abs_price_error = excluded.abs_price_error,
                    event_type = excluded.event_type,
                    event_effect_hint = excluded.event_effect_hint,
                    created_at = excluded.created_at
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def insert_drift_snapshot(self, snapshot: dict[str, Any]) -> None:
        payload_json = snapshot.get("payload_json")
        if isinstance(payload_json, (dict, list)):
            payload_json = json.dumps(payload_json)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO drift_summary
                (drift_ts, model_version, mae, mape, rmse, max_feature_psi, drift_status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["drift_ts"],
                    snapshot.get("model_version"),
                    snapshot.get("mae"),
                    snapshot.get("mape"),
                    snapshot.get("rmse"),
                    snapshot.get("max_feature_psi"),
                    snapshot.get("drift_status"),
                    payload_json,
                ),
            )
            conn.commit()

    def insert_retrain_job(self, job: dict[str, Any]) -> None:
        details_json = job.get("details_json")
        if isinstance(details_json, (dict, list)):
            details_json = json.dumps(details_json)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO retrain_jobs
                (trigger_reason, old_version, new_version, status, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.get("trigger_reason"),
                    job.get("old_version"),
                    job.get("new_version"),
                    job.get("status"),
                    details_json,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_latest_actuals_date(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(date) AS max_date FROM actuals_log").fetchone()
        return row["max_date"] if row and row["max_date"] else None

    def get_actuals_for_date(self, date: str) -> pd.DataFrame:
        return self.read_sql(
            """
            SELECT date, entity_id, demand_index, price_index, sentiment_index,
                   search_index, ad_index, event_type, event_effect_hint
            FROM actuals_log
            WHERE date = ?
            ORDER BY entity_id
            """,
            (date,),
        )

    def get_latest_actuals(self) -> pd.DataFrame:
        latest_date = self.get_latest_actuals_date()
        if not latest_date:
            return pd.DataFrame()
        return self.get_actuals_for_date(latest_date)

    def get_latest_forecast(self, limit: int = settings.FORECAST_EXPORT_LIMIT) -> pd.DataFrame:
        return self.read_sql(
            """
            SELECT
                prediction_ts,
                target_date,
                entity_id,
                pred_demand,
                pred_price,
                model_version,
                event_type,
                event_effect_hint,
                explanation_text,
                shap_explanation
            FROM predictions_log
            WHERE target_date = (SELECT MAX(target_date) FROM predictions_log)
            ORDER BY entity_id
            LIMIT ?
            """,
            (limit,),
        )

    def get_latest_drift(self) -> dict[str, Any]:
        df = self.read_sql(
            """
            SELECT drift_ts, model_version, mae, mape, rmse, max_feature_psi, drift_status, payload_json
            FROM drift_summary
            ORDER BY id DESC
            LIMIT 1
            """
        )
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_forecast_vs_actual_history(self, limit: int = settings.HISTORY_LIMIT) -> pd.DataFrame:
        return self.read_sql(
            """
            SELECT date, entity_id, pred_demand, actual_demand, demand_error, abs_demand_error,
                   pred_price, actual_price, price_error, abs_price_error, event_type,
                   event_effect_hint, model_version, created_at
            FROM forecast_vs_actual
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_runtime_state_snapshot(self) -> dict[str, Any]:
        df = self.read_sql(
            """
            SELECT state_key, state_value, updated_at
            FROM runtime_state
            ORDER BY state_key
            """
        )
        if df.empty:
            return {}
        return {
            row["state_key"]: {
                "value": row["state_value"],
                "updated_at": row["updated_at"],
            }
            for _, row in df.iterrows()
        }

    def get_recent_retrain_jobs(self, limit: int = 20) -> pd.DataFrame:
        return self.read_sql(
            """
            SELECT trigger_reason, old_version, new_version, status, details_json, created_at
            FROM retrain_jobs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )