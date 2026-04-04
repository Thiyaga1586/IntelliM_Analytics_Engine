import pandas as pd

from app.config import settings
from app.drift_manager import DriftManager
from app.forecast_manager import ForecastManager
from app.model_manager import ModelManager
from app.realtime_ingestor import RealtimeIngestor
from app.state_manager import StateManager
from app.utils import build_training_frame, read_csv_if_exists, write_csv


class AutonomousEngine:
    def __init__(self):
        self.state_manager = StateManager()
        self.model_manager = ModelManager(self.state_manager)
        self.ingestor = RealtimeIngestor(self.state_manager)
        self.forecast_manager = ForecastManager(self.state_manager, self.model_manager)
        self.drift_manager = DriftManager(self.state_manager, self.model_manager)

    def _append_master_csv(self, batch: pd.DataFrame) -> None:
        if batch.empty:
            return

        master = read_csv_if_exists(settings.MASTER_CSV_PATH)
        combined = pd.concat([master, batch], ignore_index=True)
        combined = (
            combined.sort_values(["date", "entity_id"])
            .drop_duplicates(["date", "entity_id"], keep="last")
            .reset_index(drop=True)
        )
        write_csv(combined, settings.MASTER_CSV_PATH)

    def _append_daily_summary(self, batch: pd.DataFrame) -> None:
        if batch.empty:
            return

        date_val = batch["date"].iloc[0]
        row = {
            "date": date_val,
            "avg_demand_index": float(batch["demand_index"].mean()),
            "avg_price_index": float(batch["price_index"].mean()),
            "avg_sentiment_index": float(batch["sentiment_index"].mean()),
            "avg_search_index": float(batch["search_index"].mean()),
            "avg_ad_index": float(batch["ad_index"].mean()),
            "product_count": int(batch["entity_id"].nunique()),
            "event_count": int((batch["event_type"].astype(str) != "organic").sum()),
        }

        summary = read_csv_if_exists(settings.DAILY_SUMMARY_CSV_PATH)
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary = summary.sort_values("date").drop_duplicates(["date"], keep="last")
        write_csv(summary, settings.DAILY_SUMMARY_CSV_PATH)

    def _append_alerts_from_errors(self) -> None:
        df = self.state_manager.get_forecast_vs_actual_history(limit=50)
        if df.empty:
            return

        recent = df.copy()
        recent["severity"] = recent["abs_demand_error"].apply(
            lambda x: "high" if x > 10 else "medium" if x > 5 else "low"
        )
        recent = recent[recent["severity"].isin(["high", "medium"])]

        if recent.empty:
            return

        alerts = read_csv_if_exists(settings.ALERTS_CSV_PATH)
        new_alerts = recent[["date", "entity_id", "abs_demand_error", "severity", "event_type"]].copy()
        new_alerts["alert_type"] = "forecast_miss"
        new_alerts["message"] = new_alerts.apply(
            lambda r: f"Demand deviated by {r['abs_demand_error']:.2f} for entity {r['entity_id']}",
            axis=1,
        )

        combined = pd.concat([alerts, new_alerts], ignore_index=True)
        combined = combined.drop_duplicates(["date", "entity_id", "alert_type"], keep="last")
        write_csv(combined, settings.ALERTS_CSV_PATH)

    def ingest_actuals(self, date: str | None = None) -> dict:
        batch = self.ingestor.ingest_date(date) if date else self.ingestor.ingest_next_date()
        if batch.empty:
            return {"status": "no_new_data", "rows": 0}

        self._append_master_csv(batch)
        self._append_daily_summary(batch)

        current_rows = int(self.state_manager.get_state("rows_since_last_retrain", "0"))
        self.state_manager.set_state("rows_since_last_retrain", current_rows + len(batch))

        self.drift_manager.compare_and_log(batch)
        self.drift_manager.export_forecast_vs_actual()
        drift = self.drift_manager.compute_and_store_metrics(batch)
        self._append_alerts_from_errors()

        return {
            "status": "ingested",
            "rows": int(len(batch)),
            "date": batch["date"].iloc[0],
            "drift": drift,
        }

    def refresh_forecast(self) -> dict:
        forecast_df = self.forecast_manager.refresh_forecast()
        return {
            "status": "forecast_refreshed",
            "rows": int(len(forecast_df)),
            "target_date": forecast_df["target_date"].iloc[0] if not forecast_df.empty else None,
        }

    def trigger_retrain(self, trigger_reason: str = "manual") -> dict:
        train_df = build_training_frame(self.state_manager)
        return self.model_manager.train_validate_save(
            train_df=train_df,
            trigger_reason=trigger_reason,
        )

    def maybe_retrain_or_rollback(self) -> dict:
        latest_drift = self.drift_manager.latest_status()
        if not latest_drift:
            return {"action": "none"}

        decision = self.drift_manager.evaluate_retrain_need(latest_drift)

        if decision["rollback_flag"]:
            return {"action": "rollback", "result": self.model_manager.rollback_to_last_stable()}

        if decision["should_retrain"]:
            return {"action": "retrain", "result": self.trigger_retrain(trigger_reason="policy")}

        return {"action": "none", "decision": decision}

    def run_cycle(self, date: str | None = None) -> dict:
        ingest_result = self.ingest_actuals(date=date)
        if ingest_result["status"] == "no_new_data":
            return ingest_result

        decision_result = self.maybe_retrain_or_rollback()

        forecast_result = {"status": "forecast_skipped"}
        if self.model_manager.get_active_version():
            forecast_result = self.refresh_forecast()

        return {
            "ingest": ingest_result,
            "decision": decision_result,
            "forecast": forecast_result,
        }