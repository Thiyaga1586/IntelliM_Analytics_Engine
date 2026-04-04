import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.feature_builder import FeatureBuilder
from app.model_manager import ModelManager
from app.state_manager import StateManager
from app.utils import utc_now_iso, write_csv


class DriftManager:
    def __init__(self, state_manager: StateManager, model_manager: ModelManager):
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.feature_builder = FeatureBuilder()

    def compare_and_log(self, actuals_df: pd.DataFrame) -> pd.DataFrame:
        if actuals_df.empty:
            return pd.DataFrame()

        date = actuals_df["date"].iloc[0]

        preds_df = self.state_manager.read_sql(
            """
            SELECT target_date, entity_id, pred_demand, pred_price, model_version
            FROM predictions_log
            WHERE target_date = ?
            """,
            (date,),
        )

        if preds_df.empty:
            return pd.DataFrame()

        merged = actuals_df.merge(
            preds_df,
            left_on=["date", "entity_id"],
            right_on=["target_date", "entity_id"],
            how="inner",
        )

        if merged.empty:
            return pd.DataFrame()

        merged["actual_demand"] = pd.to_numeric(merged["demand_index"], errors="coerce")
        merged["actual_price"] = pd.to_numeric(merged["price_index"], errors="coerce")
        merged["pred_demand"] = pd.to_numeric(merged["pred_demand"], errors="coerce")
        merged["pred_price"] = pd.to_numeric(merged["pred_price"], errors="coerce")

        merged = merged.dropna(
            subset=["actual_demand", "actual_price", "pred_demand", "pred_price"]
        ).copy()

        if merged.empty:
            return pd.DataFrame()

        merged["demand_error"] = merged["actual_demand"] - merged["pred_demand"]
        merged["abs_demand_error"] = merged["demand_error"].abs()
        merged["price_error"] = merged["actual_price"] - merged["pred_price"]
        merged["abs_price_error"] = merged["price_error"].abs()

        out = merged[
            [
                "date",
                "entity_id",
                "pred_demand",
                "actual_demand",
                "demand_error",
                "abs_demand_error",
                "pred_price",
                "actual_price",
                "price_error",
                "abs_price_error",
                "event_type",
                "event_effect_hint",
                "model_version",
            ]
        ].copy()

        self.state_manager.insert_forecast_vs_actual(out)
        return out

    def _load_baseline_stats(self) -> dict:
        info = self.model_manager.get_active_model_info()
        if not info.get("active_version"):
            return {}

        artifact_path = Path(info["registry_entry"]["artifact_path"])
        stats_path = artifact_path / "baseline_stats.json"

        if not stats_path.exists():
            return {}

        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_feature_psi(self, recent_actuals_df: pd.DataFrame) -> tuple[dict, float]:
        info = self.model_manager.get_active_model_info()
        if not info.get("active_version"):
            return {}, 0.0

        baseline_stats = self._load_baseline_stats()
        if not baseline_stats:
            return {}, 0.0

        feature_columns = (
            info["metadata"].get("feature_columns")
            or info["registry_entry"].get("feature_columns")
            or self.feature_builder.get_demand_feature_columns()
        )

        historical_df = self.feature_builder.load_master_training_source()
        live_feature_df = self.feature_builder.build_live_inference_frame(
            historical_df=historical_df,
            new_batch_df=recent_actuals_df,
            feature_columns=feature_columns,
        )

        if live_feature_df.empty:
            return {}, 0.0

        psi_by_feature = {}

        for col in feature_columns:
            if col not in live_feature_df.columns or col not in baseline_stats:
                continue

            values = pd.to_numeric(live_feature_df[col], errors="coerce").dropna().values
            if len(values) == 0:
                psi_by_feature[col] = 0.0
                continue

            ref = baseline_stats[col]

            # Support percentile-style training stats
            if all(k in ref for k in ["min", "q10", "q50", "q90", "max"]):
                bins = np.array(
                    [ref["min"], ref["q10"], ref["q50"], ref["q90"], ref["max"]],
                    dtype=float,
                )
                bins = np.unique(bins)

                if len(bins) < 3:
                    psi_by_feature[col] = 0.0
                    continue

                actual_hist, _ = np.histogram(values, bins=bins)
                actual_pct = actual_hist / max(actual_hist.sum(), 1)
                expected_pct = np.array([0.10, 0.40, 0.40, 0.10])[: len(actual_pct)]

            # Support mean/std baseline as fallback
            elif all(k in ref for k in ["mean", "std", "min", "max"]):
                mean_ = float(ref["mean"])
                std_ = max(float(ref["std"]), 1e-6)
                bins = np.array(
                    [
                        ref["min"],
                        mean_ - std_,
                        mean_,
                        mean_ + std_,
                        ref["max"],
                    ],
                    dtype=float,
                )
                bins = np.unique(np.sort(bins))

                if len(bins) < 3:
                    psi_by_feature[col] = 0.0
                    continue

                actual_hist, _ = np.histogram(values, bins=bins)
                actual_pct = actual_hist / max(actual_hist.sum(), 1)
                expected_pct = np.array([0.16, 0.34, 0.34, 0.16])[: len(actual_pct)]

            else:
                psi_by_feature[col] = 0.0
                continue

            actual_pct = np.clip(actual_pct, 1e-6, None)
            expected_pct = np.clip(expected_pct, 1e-6, None)

            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            psi_by_feature[col] = float(psi)

        max_psi = max(psi_by_feature.values()) if psi_by_feature else 0.0
        return psi_by_feature, max_psi

    def compute_and_store_metrics(self, recent_actuals_df: pd.DataFrame) -> dict:
        err_df = self.state_manager.read_sql(
            """
            SELECT actual_demand, pred_demand, abs_demand_error
            FROM forecast_vs_actual
            ORDER BY id DESC
            LIMIT ?
            """,
            (settings.ERROR_WINDOW,),
        )

        active_version = self.model_manager.get_active_version()

        if err_df.empty:
            payload = {
                "drift_ts": utc_now_iso(),
                "model_version": active_version,
                "mae": None,
                "mape": None,
                "rmse": None,
                "max_feature_psi": 0.0,
                "drift_status": "insufficient_data",
                "psi_by_feature": {},
            }

            self.state_manager.insert_drift_snapshot(
                {
                    "drift_ts": payload["drift_ts"],
                    "model_version": payload["model_version"],
                    "mae": None,
                    "mape": None,
                    "rmse": None,
                    "max_feature_psi": payload["max_feature_psi"],
                    "drift_status": payload["drift_status"],
                    "payload_json": json.dumps(payload),
                }
            )

            write_csv(pd.DataFrame([payload]), settings.REALTIME_DRIFT_CSV)
            return payload

        mae = float(err_df["abs_demand_error"].mean())
        mape = float(
            np.mean(
                np.abs(
                    (err_df["actual_demand"] - err_df["pred_demand"])
                    / np.clip(np.abs(err_df["actual_demand"]), 1e-6, None)
                )
            )
        )
        rmse = float(
            np.sqrt(np.mean((err_df["actual_demand"] - err_df["pred_demand"]) ** 2))
        )

        psi_by_feature, max_psi = self._compute_feature_psi(recent_actuals_df)

        baseline_mae = float(self.state_manager.get_state("active_baseline_mae", "999999"))

        if max_psi > settings.FEATURE_PSI_SEVERE_THRESHOLD:
            drift_status = "severe_feature_drift"
        elif max_psi > settings.FEATURE_PSI_THRESHOLD:
            drift_status = "feature_drift"
        elif mae > baseline_mae * settings.RETRAIN_MAE_MULTIPLIER:
            drift_status = "performance_drift"
        else:
            drift_status = "stable"

        payload = {
            "drift_ts": utc_now_iso(),
            "model_version": active_version,
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "max_feature_psi": max_psi,
            "drift_status": drift_status,
            "psi_by_feature": psi_by_feature,
        }

        self.state_manager.insert_drift_snapshot(
            {
                "drift_ts": payload["drift_ts"],
                "model_version": payload["model_version"],
                "mae": payload["mae"],
                "mape": payload["mape"],
                "rmse": payload["rmse"],
                "max_feature_psi": payload["max_feature_psi"],
                "drift_status": payload["drift_status"],
                "payload_json": json.dumps(payload),
            }
        )

        write_csv(pd.DataFrame([payload]), settings.REALTIME_DRIFT_CSV)
        return payload

    def evaluate_retrain_need(self, metrics: dict) -> dict:
        if metrics.get("drift_status") == "insufficient_data":
            return {
                "should_retrain": False,
                "rollback_flag": False,
                "performance_drift": False,
                "severe_feature_drift": False,
            }

        sim_day = int(self.state_manager.get_state("sim_day", "0"))
        last_retrain_sim_day = int(self.state_manager.get_state("last_retrain_sim_day", "0"))
        rows_since_last_retrain = int(self.state_manager.get_state("rows_since_last_retrain", "0"))
        consecutive_perf_drift = int(self.state_manager.get_state("consecutive_perf_drift", "0"))
        consecutive_rollback_drift = int(self.state_manager.get_state("consecutive_rollback_drift", "0"))
        baseline_mae = float(self.state_manager.get_state("active_baseline_mae", "999999"))

        perf_drift_now = (
            metrics["mae"] > baseline_mae * settings.RETRAIN_MAE_MULTIPLIER
            or metrics["mape"] > settings.RETRAIN_MAPE_THRESHOLD
        )

        consecutive_perf_drift = consecutive_perf_drift + 1 if perf_drift_now else 0

        rollback_drift_now = metrics["mae"] > baseline_mae * settings.ROLLBACK_MAE_MULTIPLIER
        consecutive_rollback_drift = consecutive_rollback_drift + 1 if rollback_drift_now else 0

        self.state_manager.set_state("consecutive_perf_drift", consecutive_perf_drift)
        self.state_manager.set_state("consecutive_rollback_drift", consecutive_rollback_drift)

        severe_feature_drift = metrics["max_feature_psi"] > settings.FEATURE_PSI_SEVERE_THRESHOLD
        cooldown_ok = (sim_day - last_retrain_sim_day) >= settings.RETRAIN_COOLDOWN_DAYS
        enough_rows = rows_since_last_retrain >= settings.RETRAIN_MIN_NEW_ROWS
        scheduled = (
            (sim_day - last_retrain_sim_day) >= settings.RETRAIN_SCHEDULE_DAYS
            and rows_since_last_retrain >= settings.RETRAIN_SCHEDULE_MIN_ROWS
        )

        should_retrain = cooldown_ok and (
            (perf_drift_now and consecutive_perf_drift >= 3 and enough_rows)
            or (severe_feature_drift and enough_rows)
            or scheduled
        )

        rollback_flag = consecutive_rollback_drift >= 2

        return {
            "should_retrain": should_retrain,
            "rollback_flag": rollback_flag,
            "performance_drift": perf_drift_now,
            "severe_feature_drift": severe_feature_drift,
        }

    def latest_status(self) -> dict:
        return self.state_manager.get_latest_drift()
    
    def export_forecast_vs_actual(self) -> pd.DataFrame:
        df = self.state_manager.get_forecast_vs_actual_history(limit=5000)
        if not df.empty:
            write_csv(df, settings.FORECAST_VS_ACTUAL_CSV)
        return df