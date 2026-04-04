from datetime import timedelta

import pandas as pd

from app.config import settings
from app.feature_builder import FeatureBuilder
from app.model_manager import ModelManager
from app.state_manager import StateManager
from app.utils import utc_now_iso, write_csv


class ForecastManager:
    def __init__(self, state_manager: StateManager, model_manager: ModelManager):
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.feature_builder = FeatureBuilder()

    def _generate_shap_explanations(self, feature_df: pd.DataFrame) -> list[str]:
        if feature_df.empty:
            return []

        try:
            import shap
        except Exception:
            return ["shap_unavailable"] * len(feature_df)

        model, feature_columns, _ = self.model_manager.get_active_model_bundle()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df[feature_columns])

        explanations = []
        for i in range(len(feature_df)):
            vals = shap_values[i]
            pairs = list(zip(feature_columns, vals))
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]

            lines = []
            for feat, val in pairs:
                direction = "increase" if val > 0 else "decrease"
                lines.append(f"{feat} contributed to {direction} demand")

            explanations.append(" | ".join(lines))

        return explanations

    def _generate_explanation(self, row: pd.Series) -> list[str]:
        reasons = []

        if pd.to_numeric(row.get("search_index", 0.0), errors="coerce") > 45:
            reasons.append(f"High search interest ({float(row['search_index']):.1f})")

        sentiment = pd.to_numeric(row.get("sentiment_index", 0.0), errors="coerce")
        if sentiment > 0.6:
            reasons.append("Positive sentiment trend")
        elif sentiment < 0.5:
            reasons.append("Weak sentiment affecting demand")

        if pd.to_numeric(row.get("ad_index", 0.0), errors="coerce") > 5:
            reasons.append("Strong ad campaign influence")

        event_type = str(row.get("event_type", "organic"))
        if event_type and event_type != "organic":
            reasons.append(f"Event impact: {event_type}")

        if pd.to_numeric(row.get("price_index", row.get("pred_price", 999999)), errors="coerce") < 80:
            reasons.append("Competitive pricing boosting demand")

        if not reasons:
            reasons.append("Stable organic demand")

        return reasons

    def _historical_base(self) -> pd.DataFrame:
        master_df = self.feature_builder.load_master_training_source()

        actuals_df = self.state_manager.read_sql(
            """
            SELECT date, entity_id, demand_index, price_index, sentiment_index,
                   search_index, ad_index, event_type, event_effect_hint
            FROM actuals_log
            """
        )

        frames = []
        if not master_df.empty:
            frames.append(master_df)
        if not actuals_df.empty:
            frames.append(actuals_df)

        if not frames:
            return pd.DataFrame()

        base = pd.concat(frames, ignore_index=True)
        base = (
            base.sort_values(["date", "entity_id"])
            .drop_duplicates(["date", "entity_id"], keep="last")
            .reset_index(drop=True)
        )
        return base

    def _latest_context_batch(self, history_df: pd.DataFrame) -> pd.DataFrame:
        latest_actuals = self.state_manager.get_latest_actuals()
        if not latest_actuals.empty:
            return latest_actuals

        if history_df.empty or "date" not in history_df.columns:
            return pd.DataFrame()

        latest_date = pd.to_datetime(history_df["date"], errors="coerce").max()
        if pd.isna(latest_date):
            return pd.DataFrame()

        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        batch = hist[hist["date"] == latest_date].copy()
        if batch.empty:
            return pd.DataFrame()

        batch["date"] = batch["date"].dt.strftime("%Y-%m-%d")
        return batch

    def refresh_forecast(self, base_df: pd.DataFrame = None) -> pd.DataFrame:
        active_version = self.model_manager.get_active_version()
        if not active_version:
            raise RuntimeError("No active model available. Trigger retrain first.")

        history_df = base_df if base_df is not None else self._historical_base()
        if history_df.empty:
            return pd.DataFrame()

        latest_batch = self._latest_context_batch(history_df)
        if latest_batch.empty:
            return pd.DataFrame()

        live_features = self.feature_builder.build_live_inference_frame(
            historical_df=history_df,
            new_batch_df=latest_batch,
        )

        if live_features.empty:
            return pd.DataFrame()

        current_date = pd.to_datetime(latest_batch["date"].max())
        next_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

        pred_series = self.model_manager.predict_batch(
            live_features.drop(columns=["entity_id", "date"])
        )
        shap_explanations = self._generate_shap_explanations(
            live_features.drop(columns=["entity_id", "date"])
        )

        out = live_features[["entity_id"]].copy()
        out["prediction_ts"] = utc_now_iso()
        out["target_date"] = next_date
        out["pred_demand"] = pred_series.values

        keep_cols = [
            "entity_id",
            "price_index",
            "search_index",
            "sentiment_index",
            "ad_index",
            "event_type",
            "event_effect_hint",
        ]
        for extra in ["brand", "product_name", "category"]:
            if extra in latest_batch.columns:
                keep_cols.append(extra)

        latest_actuals_map = (
            latest_batch[keep_cols]
            .drop_duplicates("entity_id")
            .rename(columns={"price_index": "pred_price"})
        )

        out = out.merge(latest_actuals_map, on="entity_id", how="left")
        out["pred_price"] = pd.to_numeric(out["pred_price"], errors="coerce").fillna(0.0)
        out["pred_demand"] = pd.to_numeric(out["pred_demand"], errors="coerce").fillna(0.0)
        out["model_version"] = active_version

        out["explanation"] = out.apply(self._generate_explanation, axis=1)
        out["explanation_text"] = out["explanation"].apply(lambda x: " | ".join(x))
        out["shap_explanation"] = shap_explanations

        final_cols = [
            "prediction_ts",
            "target_date",
            "entity_id",
            "pred_demand",
            "pred_price",
            "model_version",
            "search_index",
            "sentiment_index",
            "ad_index",
            "event_type",
            "event_effect_hint",
            "explanation_text",
            "shap_explanation",
        ]
        for extra in ["brand", "product_name", "category"]:
            if extra in out.columns:
                final_cols.append(extra)

        out = out[final_cols].copy()

        self.state_manager.insert_predictions(out)
        write_csv(out, settings.APP_FORECAST_CSV)
        return out

    def get_latest_forecast(self) -> pd.DataFrame:
        df = self.state_manager.get_latest_forecast()
        if not df.empty:
            write_csv(df, settings.APP_FORECAST_CSV)
        return df