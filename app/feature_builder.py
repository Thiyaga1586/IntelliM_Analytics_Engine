import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.utils import read_csv_if_exists


class FeatureBuilder:
    def __init__(self, feature_metadata_path: Optional[Path] = None):
        self.feature_metadata_path = feature_metadata_path or settings.FEATURE_METADATA_PATH
        self.feature_metadata = self._load_feature_metadata()

    def validate_feature_contract(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        feature_columns = feature_columns or self.get_demand_feature_columns()
        missing = [c for c in feature_columns if c not in df.columns]
        present = [c for c in feature_columns if c in df.columns]
        return {"missing": missing, "present": present}
    
    def build_debug_frames(
        self,
        historical_df: pd.DataFrame,
        new_batch_df: Optional[pd.DataFrame] = None,
        forecast_horizon: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        events_df = self.load_events_source()
        regime_df = self.load_regime_source()

        full_train = self._build_full_feature_frame(historical_df, events_df, regime_df).copy()
        final_train = self.build_training_frame(
            historical_df=historical_df,
            forecast_horizon=forecast_horizon,
        )

        result = {
            "full_training_frame": full_train,
            "final_training_frame": final_train,
        }

        if new_batch_df is not None and not new_batch_df.empty:
            hist = self._normalize_base_columns(historical_df)
            batch = self._normalize_base_columns(new_batch_df)

            all_df = pd.concat([hist, batch], ignore_index=True)
            all_df = all_df.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date"], keep="last").reset_index(drop=True)

            full_live = self._build_full_feature_frame(all_df, events_df, regime_df).copy()
            final_live = self.build_live_inference_frame(
                historical_df=historical_df,
                new_batch_df=new_batch_df,
            )

            result["full_live_frame"] = full_live
            result["final_live_frame"] = final_live

        return result

    def _load_feature_metadata(self) -> Dict:
        if not self.feature_metadata_path.exists():
            raise FileNotFoundError(f"feature_metadata.json not found at: {self.feature_metadata_path}")
        with open(self.feature_metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_demand_feature_columns(self) -> List[str]:
        base_cols = list(self.feature_metadata["demand_model"]["feature_names"])

        phase2_extra_cols = [
            "demand_lag_1",
            "demand_lag_2",
            "demand_lag_3",
            "demand_lag_7",
            "demand_3d_mean",
            "demand_7d_mean",
            "demand_7d_std",
            "demand_change_1",
            "demand_change_3",
            "demand_momentum_7",
            "demand_volatility_7d",
            "demand_range_7d",
            "trend_strength_7d",
            "event_flag_today",
            "event_flag_recent_3d",
            "event_flag_next_3d",
            "days_since_last_event",
            "days_to_next_event",
            "event_intensity_score",
            "event_type_encoded",
            "recent_change_point_flag",
            "regime_shift_strength",
            "regime_flag_recent",
            "price_x_search",
            "search_x_sentiment",
            "ad_x_search",
            "discount_x_demand_lag_1",
            "buzz_x_search",
            "rating_x_reviews",
            "is_weekend",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        ]

        merged = []
        seen = set()

        for col in base_cols + phase2_extra_cols:
            if col not in seen:
                merged.append(col)
                seen.add(col)

        return merged

    def get_demand_target(self) -> str:
        return self.feature_metadata["demand_model"]["target"]

    def load_master_training_source(self) -> pd.DataFrame:
        return read_csv_if_exists(settings.MASTER_CSV_PATH)

    def load_events_source(self) -> pd.DataFrame:
        return read_csv_if_exists(settings.EVENTS_CSV_PATH)

    def load_regime_source(self) -> pd.DataFrame:
        return read_csv_if_exists(settings.REGIME_SHIFTS_CSV_PATH)

    def _normalize_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        out = df.copy()

        rename_map = {
            "product_id": "entity_id",
            "sku": "entity_id",
            "asin": "entity_id",
        }
        out = out.rename(columns=rename_map)

        if "entity_id" not in out.columns:
            raise ValueError("Missing entity_id (or product_id / sku / asin)")

        if "date" not in out.columns:
            raise ValueError("Missing date")

        out["entity_id"] = out["entity_id"].astype(str)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "entity_id"]).copy()

        alias_rules = {
            "price_index": "price_new",
            "search_index": "search_interest",
            "sentiment_index": "sentiment_mean",
        }
        for src, dst in alias_rules.items():
            if dst not in out.columns and src in out.columns:
                out[dst] = out[src]

        fill_defaults = {
            "new_reviews": 0.0,
            "avg_rating": 4.0,
            "sentiment_mean": 0.0,
            "search_interest": 0.0,
            "buzz_score": 0.0,
            "stock_status_score": 1.0,
            "listing_change_score": 0.0,
            "discount_pct": 0.0,
            "rank_proxy": 0.0,
            "bsr_category_rank": 0.0,
            "price_competitiveness_index": 0.0,
            "competitor_count": 0.0,
            "price_new": 0.0,
            "ad_index": 0.0,
            "search_index": 0.0,
            "sentiment_index": 0.0,
            "health_index": 0.0,
            "event_type": "",
            "event_effect_hint": "",
        }

        for col, default in fill_defaults.items():
            if col not in out.columns:
                out[col] = default

        numeric_cols = [
            "new_reviews",
            "avg_rating",
            "sentiment_mean",
            "search_interest",
            "buzz_score",
            "stock_status_score",
            "listing_change_score",
            "discount_pct",
            "rank_proxy",
            "bsr_category_rank",
            "price_competitiveness_index",
            "competitor_count",
            "price_new",
            "ad_index",
            "search_index",
            "sentiment_index",
            "health_index",
        ]

        if "demand_index" in out.columns:
            numeric_cols.append("demand_index")

        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        for col, default in fill_defaults.items():
            if col in numeric_cols:
                out[col] = out[col].fillna(default)
            else:
                out[col] = out[col].fillna(default)

        out = out.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date"], keep="last").reset_index(drop=True)
        return out

    def _normalize_events_columns(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty:
            return pd.DataFrame(columns=["entity_id", "date", "event_type", "event_intensity_score"])

        out = events_df.copy()

        rename_map = {
            "product_id": "entity_id",
            "sku": "entity_id",
            "asin": "entity_id",
            "event_date": "date",
            "start_date": "date",
        }
        out = out.rename(columns=rename_map)

        if "entity_id" not in out.columns:
            return pd.DataFrame(columns=["entity_id", "date", "event_type", "event_intensity_score"])
        if "date" not in out.columns:
            return pd.DataFrame(columns=["entity_id", "date", "event_type", "event_intensity_score"])

        out["entity_id"] = out["entity_id"].astype(str)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["entity_id", "date"]).copy()

        if "event_type" not in out.columns:
            out["event_type"] = "generic_event"

        intensity_candidates = ["event_intensity_score", "event_intensity", "impact_score", "weight"]
        intensity_col = next((c for c in intensity_candidates if c in out.columns), None)
        if intensity_col is None:
            out["event_intensity_score"] = 1.0
        else:
            out["event_intensity_score"] = pd.to_numeric(out[intensity_col], errors="coerce").fillna(1.0)

        out = out[["entity_id", "date", "event_type", "event_intensity_score"]].copy()
        out = out.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date", "event_type"], keep="last")
        return out

    def _normalize_regime_columns(self, regime_df: pd.DataFrame) -> pd.DataFrame:
        if regime_df.empty:
            return pd.DataFrame(columns=["entity_id", "date", "recent_change_point_flag", "regime_shift_strength"])

        out = regime_df.copy()

        rename_map = {
            "product_id": "entity_id",
            "sku": "entity_id",
            "asin": "entity_id",
        }
        out = out.rename(columns=rename_map)

        if "entity_id" not in out.columns or "date" not in out.columns:
            return pd.DataFrame(columns=["entity_id", "date", "recent_change_point_flag", "regime_shift_strength"])

        out["entity_id"] = out["entity_id"].astype(str)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["entity_id", "date"]).copy()

        cp_col = None
        for c in ["change_point", "rule_change_point", "is_change_point"]:
            if c in out.columns:
                cp_col = c
                break

        shift_col = None
        for c in ["shift_strength", "regime_shift_strength"]:
            if c in out.columns:
                shift_col = c
                break

        out["recent_change_point_flag"] = pd.to_numeric(out[cp_col], errors="coerce").fillna(0.0) if cp_col else 0.0
        out["regime_shift_strength"] = pd.to_numeric(out[shift_col], errors="coerce").fillna(0.0) if shift_col else 0.0

        out = out[["entity_id", "date", "recent_change_point_flag", "regime_shift_strength"]].copy()
        out = out.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date"], keep="last")
        return out

    def _add_autoregressive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grp = out.groupby("entity_id", group_keys=False)

        if "demand_index" not in out.columns:
            out["demand_index"] = np.nan

        out["demand_lag_1"] = grp["demand_index"].shift(1)
        out["demand_lag_2"] = grp["demand_index"].shift(2)
        out["demand_lag_3"] = grp["demand_index"].shift(3)
        out["demand_lag_7"] = grp["demand_index"].shift(7)

        out["demand_3d_mean"] = grp["demand_index"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        out["demand_7d_mean"] = grp["demand_index"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        out["demand_7d_std"] = grp["demand_index"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())

        out["demand_change_1"] = out["demand_lag_1"] - out["demand_lag_2"]
        out["demand_change_3"] = out["demand_lag_1"] - out["demand_lag_3"]
        out["demand_momentum_7"] = out["demand_lag_1"] - out["demand_lag_7"]

        out["demand_volatility_7d"] = grp["demand_index"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        out["demand_range_7d"] = grp["demand_index"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).max()) - \
                                 grp["demand_index"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).min())
        out["trend_strength_7d"] = out["demand_lag_1"] - out["demand_7d_mean"]

        return out

    def _add_market_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        grp = out.groupby("entity_id", group_keys=False)

        out["reviews_7d_sum"] = grp["new_reviews"].transform(lambda s: s.rolling(7, min_periods=1).sum())
        out["search_7d_avg"] = grp["search_interest"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["sentiment_7d_avg"] = grp["sentiment_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["price_7d_avg"] = grp["price_new"].transform(lambda s: s.rolling(7, min_periods=1).mean())

        lag_base_cols = ["new_reviews", "search_interest", "sentiment_mean", "price_new", "avg_rating", "buzz_score"]
        for col in lag_base_cols:
            out[f"{col}_lag_1"] = grp[col].shift(1)
            out[f"{col}_lag_7"] = grp[col].shift(7)

        out["new_reviews_7d_sum"] = grp["new_reviews"].transform(lambda s: s.rolling(7, min_periods=1).sum())
        out["new_reviews_7d_mean"] = grp["new_reviews"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["search_interest_7d_mean"] = grp["search_interest"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["sentiment_mean_7d_mean"] = grp["sentiment_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["price_new_7d_mean"] = grp["price_new"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["price_new_7d_std"] = grp["price_new"].transform(lambda s: s.rolling(7, min_periods=2).std())
        out["buzz_score_7d_mean"] = grp["buzz_score"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["avg_rating_7d_mean"] = grp["avg_rating"].transform(lambda s: s.rolling(7, min_periods=1).mean())

        change_cols = ["new_reviews", "search_interest", "sentiment_mean", "price_new", "avg_rating", "buzz_score"]
        for col in change_cols:
            out[f"{col}_change"] = grp[col].transform(lambda s: s.diff())
            out[f"{col}_momentum"] = grp[col].transform(lambda s: s.diff(3))

        needs_health_fill = out["health_index"].isna() | (out["health_index"] == 0)
        out.loc[needs_health_fill, "health_index"] = (
            0.30 * out.loc[needs_health_fill, "sentiment_index"].fillna(0.0)
            + 0.25 * out.loc[needs_health_fill, "search_index"].fillna(0.0)
            + 0.20 * out.loc[needs_health_fill, "ad_index"].fillna(0.0)
            + 0.15 * out.loc[needs_health_fill, "avg_rating"].fillna(4.0)
            + 0.10 * (1.0 - out.loc[needs_health_fill, "discount_pct"].fillna(0.0))
        )

        return out

    def _merge_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        events = self._normalize_events_columns(events_df)

        out["event_flag_today"] = 0.0
        out["event_flag_recent_3d"] = 0.0
        out["event_flag_next_3d"] = 0.0
        out["days_since_last_event"] = 999.0
        out["days_to_next_event"] = 999.0
        out["event_intensity_score"] = 0.0
        out["event_type_encoded"] = 0.0

        if events.empty or out.empty:
            return out

        event_type_map = {name: idx + 1 for idx, name in enumerate(sorted(events["event_type"].astype(str).unique().tolist()))}
        events["event_type_encoded"] = events["event_type"].astype(str).map(event_type_map).fillna(0.0)

        event_groups = {
            entity: grp.sort_values("date").reset_index(drop=True)
            for entity, grp in events.groupby("entity_id")
        }

        def row_event_features(row: pd.Series) -> Tuple[float, float, float, float, float, float, float]:
            entity = row["entity_id"]
            current_date = row["date"]

            entity_events = event_groups.get(entity)
            if entity_events is None or entity_events.empty:
                return 0.0, 0.0, 0.0, 999.0, 999.0, 0.0, 0.0

            deltas = (entity_events["date"] - current_date).dt.days

            today_mask = deltas == 0
            recent_mask = (deltas >= -3) & (deltas <= 0)
            next_mask = (deltas >= 0) & (deltas <= 3)
            past_mask = deltas <= 0
            future_mask = deltas >= 0

            event_flag_today = 1.0 if today_mask.any() else 0.0
            event_flag_recent_3d = 1.0 if recent_mask.any() else 0.0
            event_flag_next_3d = 1.0 if next_mask.any() else 0.0

            days_since_last_event = float(abs(deltas[past_mask].max())) if past_mask.any() else 999.0
            days_to_next_event = float(deltas[future_mask].min()) if future_mask.any() else 999.0

            local_window = entity_events[(deltas >= -3) & (deltas <= 3)]
            event_intensity_score = float(local_window["event_intensity_score"].max()) if not local_window.empty else 0.0
            event_type_encoded = float(local_window["event_type_encoded"].max()) if not local_window.empty else 0.0

            return (
                event_flag_today,
                event_flag_recent_3d,
                event_flag_next_3d,
                days_since_last_event,
                days_to_next_event,
                event_intensity_score,
                event_type_encoded,
            )

        event_features = out.apply(row_event_features, axis=1, result_type="expand")
        event_features.columns = [
            "event_flag_today",
            "event_flag_recent_3d",
            "event_flag_next_3d",
            "days_since_last_event",
            "days_to_next_event",
            "event_intensity_score",
            "event_type_encoded",
        ]

        out[
            [
                "event_flag_today",
                "event_flag_recent_3d",
                "event_flag_next_3d",
                "days_since_last_event",
                "days_to_next_event",
                "event_intensity_score",
                "event_type_encoded",
            ]
        ] = event_features

        return out

    def _merge_regime_features(self, df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        regime = self._normalize_regime_columns(regime_df)

        out["recent_change_point_flag"] = 0.0
        out["regime_shift_strength"] = 0.0
        out["regime_flag_recent"] = 0.0

        if regime.empty or out.empty:
            return out

        out = out.merge(regime, on=["entity_id", "date"], how="left", suffixes=("", "_reg"))

        if "recent_change_point_flag_reg" in out.columns:
            out["recent_change_point_flag"] = out["recent_change_point_flag_reg"].fillna(out["recent_change_point_flag"])
            out = out.drop(columns=["recent_change_point_flag_reg"])

        if "regime_shift_strength_reg" in out.columns:
            out["regime_shift_strength"] = out["regime_shift_strength_reg"].fillna(out["regime_shift_strength"])
            out = out.drop(columns=["regime_shift_strength_reg"])

        grp = out.groupby("entity_id", group_keys=False)
        out["regime_flag_recent"] = grp["recent_change_point_flag"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).max()).fillna(0.0)
        return out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["price_x_search"] = out["price_new"] * out["search_interest"]
        out["search_x_sentiment"] = out["search_interest"] * out["sentiment_mean"]
        out["ad_x_search"] = out["ad_index"] * out["search_interest"]
        out["discount_x_demand_lag_1"] = out["discount_pct"] * out["demand_lag_1"].fillna(0.0)
        out["buzz_x_search"] = out["buzz_score"] * out["search_interest"]
        out["rating_x_reviews"] = out["avg_rating"] * out["new_reviews"]

        return out

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["day_of_week_num"] = out["date"].dt.dayofweek
        out["month_num"] = out["date"].dt.month
        out["is_weekend"] = out["day_of_week_num"].isin([5, 6]).astype(float)

        out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week_num"] / 7.0)
        out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week_num"] / 7.0)

        out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)

        return out

    def _finalize_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        out = df.copy()

        for col in feature_columns:
            if col not in out.columns:
                out[col] = 0.0

        out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
        out[feature_columns] = out[feature_columns].fillna(0.0)

        return out

    def _build_full_feature_frame(self, base_df: pd.DataFrame, events_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
        out = self._normalize_base_columns(base_df)
        out = self._add_autoregressive_features(out)
        out = self._add_market_signal_features(out)
        out = self._merge_event_features(out, events_df)
        out = self._merge_regime_features(out, regime_df)
        out = self._add_interaction_features(out)
        out = self._add_calendar_features(out)
        return out

    def build_training_frame(
        self,
        historical_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        forecast_horizon: int = 1,
    ) -> pd.DataFrame:
        feature_columns = feature_columns or self.get_demand_feature_columns()
        target_col = target_col or self.get_demand_target()

        events_df = self.load_events_source()
        regime_df = self.load_regime_source()

        base = self._build_full_feature_frame(historical_df, events_df, regime_df)
        grp = base.groupby("entity_id", group_keys=False)

        future_target_col = f"{target_col}_t_plus_{forecast_horizon}"
        base[future_target_col] = grp[target_col].shift(-forecast_horizon)

        base = self._finalize_features(base, feature_columns)

        cols = ["entity_id", "date"] + feature_columns + [future_target_col]
        base = base[cols].copy()
        base = base.dropna(subset=[future_target_col]).sort_values(["date", "entity_id"]).reset_index(drop=True)
        base = base.rename(columns={future_target_col: target_col})
        return base

    def build_live_inference_frame(
        self,
        historical_df: pd.DataFrame,
        new_batch_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        feature_columns = feature_columns or self.get_demand_feature_columns()

        hist = self._normalize_base_columns(historical_df)
        batch = self._normalize_base_columns(new_batch_df)

        if batch.empty:
            return pd.DataFrame()

        all_df = pd.concat([hist, batch], ignore_index=True)
        all_df = all_df.sort_values(["entity_id", "date"]).drop_duplicates(["entity_id", "date"], keep="last").reset_index(drop=True)

        events_df = self.load_events_source()
        regime_df = self.load_regime_source()

        all_df = self._build_full_feature_frame(all_df, events_df, regime_df)
        all_df = self._finalize_features(all_df, feature_columns)

        target_dates = sorted(batch["date"].unique().tolist())
        infer = all_df[all_df["date"].isin(target_dates)].copy()
        infer = infer.sort_values(["date", "entity_id"]).reset_index(drop=True)

        cols = ["entity_id", "date"] + feature_columns
        return infer[cols].copy()
    
if __name__ == "__main__":
    from app.utils import read_csv_if_exists
    from app.config import settings

    fb = FeatureBuilder()

    master_df = fb.load_master_training_source()
    query_df = read_csv_if_exists(settings.QUERY_CSV_PATH)

    print("Master shape:", master_df.shape)
    print("Query shape:", query_df.shape)

    debug_frames = fb.build_debug_frames(
        historical_df=master_df,
        new_batch_df=query_df,
        forecast_horizon=1,
    )

    full_train = debug_frames["full_training_frame"]
    final_train = debug_frames["final_training_frame"]

    print("Full training frame shape:", full_train.shape)
    print("Final training frame shape:", final_train.shape)

    contract_report = fb.validate_feature_contract(final_train.drop(columns=["entity_id", "date", fb.get_demand_target()], errors="ignore"))
    print("Missing feature columns in final training frame:", contract_report["missing"])

    full_train_path = settings.OUTPUT_DIR / "debug_full_training_frame.csv"
    final_train_path = settings.OUTPUT_DIR / "debug_training_features.csv"

    full_train.to_csv(full_train_path, index=False)
    final_train.to_csv(final_train_path, index=False)

    print(f"Saved full training frame → {full_train_path}")
    print(f"Saved final training features → {final_train_path}")

    if not query_df.empty:
        full_live = debug_frames["full_live_frame"]
        final_live = debug_frames["final_live_frame"]

        print("Full live frame shape:", full_live.shape)
        print("Final live frame shape:", final_live.shape)

        full_live_path = settings.OUTPUT_DIR / "debug_full_live_frame.csv"
        final_live_path = settings.OUTPUT_DIR / "debug_live_features.csv"

        full_live.to_csv(full_live_path, index=False)
        final_live.to_csv(final_live_path, index=False)

        print(f"Saved full live frame → {full_live_path}")
        print(f"Saved final live features → {final_live_path}")
    else:
        print("query.csv is empty or missing")