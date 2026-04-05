from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.utils import read_csv_if_exists


@dataclass
class DashboardBundleBuilder:
    weeks_default: int = 12
    days_default: int = 7

    def __post_init__(self) -> None:
        self.color_palette = [
            "#C9A84C",
            "#1A3A5C",
            "#22C55E",
            "#5B8DB8",
            "#FF6B35",
            "#2C2C2C",
            "#FF4444",
            "#00CED1",
            "#9B59B6",
            "#FF8800",
            "#FF1493",
        ]

    # -----------------------------
    # Public entrypoint
    # -----------------------------
    def build(self, category: str = "all", weeks: int = 12, days: int = 7) -> dict[str, Any]:
        weeks = weeks or self.weeks_default
        days = days or self.days_default

        master = self._load_master()
        alerts = self._load_alerts()
        forecast = self._load_forecast()
        fva = self._load_forecast_vs_actual()
        events = self._load_events()
        regime = self._load_regime()

        selected_master = self._filter_category(master, category)
        selected_fva = self._filter_category(fva, category)

        return {
            "kpis": self._build_kpis(master, alerts),
            "w12": self._build_w12(selected_master, weeks),
            "w12_pred": self._build_w12_pred(selected_fva, weeks),
            "w12_by_cat": self._build_w12_by_cat(master, weeks),
            "w12_by_cat_pred": self._build_w12_by_cat_pred(fva, weeks),
            "next_days": self._build_next_days(fva, forecast, days, category),
            "kpis_by_cat": self._build_kpis_by_cat(master),
            "scatter": self._build_scatter(master),
            "radar": self._build_radar(master, category="all"),
            "radar_by_cat": self._build_radar_by_cat(master),
            "trends_data": self._build_trends_data(master),
            "competitors": self._build_competitors(master),
            "events": self._build_events(events, category),
            "buzz": self._build_buzz(master, category="all"),
            "earbuds_s": self._build_category_sparkline(master, "earbuds"),
            "phones_s": self._build_category_sparkline(master, "smartphones"),
            "regime": self._build_regime(regime),
            "report": self._build_report(master, alerts),
        }

    # -----------------------------
    # Loaders
    # -----------------------------
    def _load_master(self) -> pd.DataFrame:
        df = read_csv_if_exists(settings.MASTER_CSV_PATH)
        if df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in [
            "demand_index",
            "price_index",
            "sentiment_index",
            "search_index",
            "ad_index",
            "buzz_score",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["entity_id", "brand", "product_name", "category"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df.dropna(subset=["date", "entity_id"])
        return df

    def _load_alerts(self) -> pd.DataFrame:
        return read_csv_if_exists(settings.ALERTS_CSV_PATH)

    def _load_forecast(self) -> pd.DataFrame:
        df = read_csv_if_exists(settings.APP_FORECAST_CSV)
        if df.empty:
            return df

        df = df.copy()
        if "target_date" in df.columns:
            df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")

        for col in ["pred_demand", "pred_price", "search_index", "sentiment_index", "ad_index"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "entity_id" in df.columns:
            df["entity_id"] = df["entity_id"].astype(str)

        return df

    def _load_forecast_vs_actual(self) -> pd.DataFrame:
        df = read_csv_if_exists(settings.FORECAST_VS_ACTUAL_CSV)
        if df.empty:
            return df

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in [
            "pred_demand",
            "actual_demand",
            "pred_price",
            "actual_price",
            "demand_error",
            "abs_demand_error",
            "price_error",
            "abs_price_error",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "entity_id" in df.columns:
            df["entity_id"] = df["entity_id"].astype(str)

        # enrich category/brand/product from master if missing
        master = self._load_master()
        if not master.empty and "entity_id" in df.columns:
            lookup = (
                master.sort_values("date")
                .drop_duplicates("entity_id", keep="last")[["entity_id", "brand", "product_name", "category"]]
            )
            df = df.merge(lookup, on="entity_id", how="left")

        return df

    def _load_events(self) -> pd.DataFrame:
        df = read_csv_if_exists(settings.EVENTS_CSV_PATH)
        if df.empty:
            return df

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["entity_id", "brand", "product_name", "category"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _load_regime(self) -> pd.DataFrame:
        df = read_csv_if_exists(settings.REGIME_SHIFTS_CSV_PATH)
        if df.empty:
            return df

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    # -----------------------------
    # Helpers
    # -----------------------------
    def _filter_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        if df.empty or category == "all":
            return df
        if "category" not in df.columns:
            return df.iloc[0:0].copy()
        return df[df["category"].astype(str).str.lower() == str(category).lower()].copy()

    def _latest_two_dates(self, df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if df.empty or "date" not in df.columns:
            return None, None
        dates = sorted(df["date"].dropna().unique().tolist())
        if not dates:
            return None, None
        latest = dates[-1]
        prev = dates[-2] if len(dates) >= 2 else None
        return latest, prev

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _pct_change(self, current: float, previous: float) -> float:
        if previous is None or abs(previous) < 1e-9:
            return 0.0
        return round(((current - previous) / previous) * 100.0, 1)

    def _point_change(self, current: float, previous: float) -> float:
        return round(current - previous, 1)

    def _norm_to_100(self, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all():
            return pd.Series([50.0] * len(s), index=s.index)
        min_v = s.min()
        max_v = s.max()
        if pd.isna(min_v) or pd.isna(max_v) or abs(max_v - min_v) < 1e-9:
            return pd.Series([50.0] * len(s), index=s.index)
        return ((s - min_v) / (max_v - min_v) * 100.0).clip(0, 100)

    def _week_series(self, df: pd.DataFrame, value_col: str, weeks: int) -> list[float]:
        if df.empty or value_col not in df.columns or "date" not in df.columns:
            return []

        temp = df.dropna(subset=["date"]).copy()
        if temp.empty:
            return []

        temp["week"] = temp["date"].dt.to_period("W").astype(str)
        series = temp.groupby("week")[value_col].mean().tail(weeks)
        return [round(self._safe_float(v), 2) for v in series.tolist()]

    def _week_series_by_cat(self, df: pd.DataFrame, value_col: str, weeks: int, cats: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if df.empty or "category" not in df.columns:
            return out

        for cat in cats:
            sub = df[df["category"].astype(str).str.lower() == cat.lower()].copy()
            out[cat] = self._week_series(sub, value_col, weeks)
        return out

    def _daily_series(self, df: pd.DataFrame, value_col: str, n: int) -> list[float]:
        if df.empty or value_col not in df.columns or "date" not in df.columns:
            return []

        temp = df.dropna(subset=["date"]).copy()
        series = temp.groupby(temp["date"].dt.strftime("%Y-%m-%d"))[value_col].mean().tail(n)
        return [round(self._safe_float(v), 2) for v in series.tolist()]

    def _latest_by_brand(self, master: pd.DataFrame) -> pd.DataFrame:
        if master.empty:
            return master
        latest_date, _ = self._latest_two_dates(master)
        if latest_date is None:
            return master.iloc[0:0].copy()
        latest = master[master["date"] == latest_date].copy()
        return latest

    def _top_categories(self, master: pd.DataFrame, n: int = 3) -> list[str]:
        if master.empty or "category" not in master.columns:
            return []
        latest = self._latest_by_brand(master)
        if latest.empty:
            latest = master
        cats = (
            latest.groupby("category")["demand_index"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
            .index.astype(str)
            .tolist()
        )
        return cats

    def _brand_color_map(self, brands: list[str]) -> dict[str, str]:
        return {brand: self.color_palette[i % len(self.color_palette)] for i, brand in enumerate(brands)}

    # -----------------------------
    # Builders
    # -----------------------------
    def _build_kpis(self, master: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, Any]:
        if master.empty:
            return {
                "demand": 0.0,
                "demand_chg": 0.0,
                "price": 0.0,
                "price_chg": 0.0,
                "sentiment": 0.0,
                "sent_chg": 0.0,
                "active_brands": 0,
                "active_entities": 0,
                "total_rows": 0,
                "total_alerts": 0,
            }

        latest_date, prev_date = self._latest_two_dates(master)
        latest = master[master["date"] == latest_date].copy()
        prev = master[master["date"] == prev_date].copy() if prev_date is not None else pd.DataFrame()

        demand = round(self._safe_float(latest["demand_index"].mean()), 1)
        price = round(self._safe_float(latest["price_index"].mean()), 2)
        sentiment = round(self._safe_float(latest["sentiment_index"].mean()) * 100.0, 1)

        prev_demand = self._safe_float(prev["demand_index"].mean(), demand)
        prev_price = self._safe_float(prev["price_index"].mean(), price)
        prev_sent = self._safe_float(prev["sentiment_index"].mean(), sentiment / 100.0) * 100.0

        return {
            "demand": demand,
            "demand_chg": self._pct_change(demand, prev_demand),
            "price": price,
            "price_chg": self._pct_change(price, prev_price),
            "sentiment": sentiment,
            "sent_chg": self._point_change(sentiment, prev_sent),
            "active_brands": int(master["brand"].nunique()) if "brand" in master.columns else 0,
            "active_entities": int(master["entity_id"].nunique()) if "entity_id" in master.columns else 0,
            "total_rows": int(len(master)),
            "total_alerts": int(len(alerts)),
        }

    def _build_w12(self, master: pd.DataFrame, weeks: int) -> list[float]:
        return self._week_series(master, "demand_index", weeks)

    def _build_w12_pred(self, fva: pd.DataFrame, weeks: int) -> list[float]:
        if fva.empty or "pred_demand" not in fva.columns:
            return []
        return self._week_series(fva.rename(columns={"date": "date"}), "pred_demand", weeks)

    def _build_w12_by_cat(self, master: pd.DataFrame, weeks: int) -> dict[str, list[float]]:
        cats = self._top_categories(master, n=3)
        return self._week_series_by_cat(master, "demand_index", weeks, cats)

    def _build_w12_by_cat_pred(self, fva: pd.DataFrame, weeks: int) -> dict[str, list[float]]:
        if fva.empty:
            return {}
        cats = self._top_categories(fva.rename(columns={"actual_demand": "demand_index"}), n=3)
        return self._week_series_by_cat(fva, "pred_demand", weeks, cats)

    def _build_next_days(self, fva: pd.DataFrame, forecast: pd.DataFrame, days: int, category: str) -> list[dict[str, Any]]:
        selected_fva = self._filter_category(fva, category)
        if not selected_fva.empty and {"date", "actual_demand", "pred_demand"}.issubset(selected_fva.columns):
            temp = selected_fva.copy()
            temp["label"] = temp["date"].dt.strftime("%b %-d") if hasattr(temp["date"].dt, "strftime") else temp["date"].astype(str)
            grouped = (
                temp.groupby(temp["date"].dt.strftime("%b %-d"))[["actual_demand", "pred_demand"]]
                .mean()
                .tail(days)
                .reset_index()
            )
            return [
                {
                    "label": row["date"],
                    "actual": round(self._safe_float(row["actual_demand"]), 2),
                    "predicted": round(self._safe_float(row["pred_demand"]), 2),
                }
                for _, row in grouped.iterrows()
            ]

        selected_forecast = self._filter_category(forecast, category)
        if selected_forecast.empty or "target_date" not in selected_forecast.columns:
            return []

        grouped = (
            selected_forecast.groupby(selected_forecast["target_date"].dt.strftime("%b %-d"))["pred_demand"]
            .mean()
            .tail(days)
            .reset_index()
        )
        return [
            {
                "label": row["target_date"],
                "actual": None,
                "predicted": round(self._safe_float(row["pred_demand"]), 2),
            }
            for _, row in grouped.iterrows()
        ]

    def _build_kpis_by_cat(self, master: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if master.empty or "category" not in master.columns:
            return {}

        latest_date, _ = self._latest_two_dates(master)
        latest = master[master["date"] == latest_date].copy() if latest_date is not None else master.copy()

        out: dict[str, dict[str, Any]] = {}
        for cat, grp in latest.groupby("category"):
            out[str(cat)] = {
                "demand": round(self._safe_float(grp["demand_index"].mean()), 1),
                "price": round(self._safe_float(grp["price_index"].mean()), 1),
                "sentiment": round(self._safe_float(grp["sentiment_index"].mean()) * 100.0, 1),
                "count": int(grp["entity_id"].nunique()),
            }
        return out

    def _build_scatter(self, master: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        if master.empty or "brand" not in master.columns:
            return {"all": []}

        latest_date, _ = self._latest_two_dates(master)
        latest = master[master["date"] == latest_date].copy() if latest_date is not None else master.copy()

        def build_for(df: pd.DataFrame) -> list[dict[str, Any]]:
            if df.empty:
                return []
            grp = (
                df.groupby("brand")
                .agg(
                    x=("price_index", "mean"),
                    y=("demand_index", "mean"),
                )
                .reset_index()
                .sort_values("y", ascending=False)
                .head(14)
            )
            return [
                {
                    "x": round(self._safe_float(row["x"]), 1),
                    "y": round(self._safe_float(row["y"]), 1),
                    "label": str(row["brand"]),
                }
                for _, row in grp.iterrows()
            ]

        cats = self._top_categories(master, n=3)
        out = {"all": build_for(latest)}
        for cat in cats:
            out[cat] = build_for(latest[latest["category"].astype(str).str.lower() == cat.lower()].copy())
        return out

    def _build_radar(self, master: pd.DataFrame, category: str = "all") -> list[dict[str, Any]]:
        selected = self._filter_category(master, category)
        if selected.empty or "brand" not in selected.columns:
            return []

        latest = self._latest_by_brand(selected)
        if latest.empty:
            latest = selected.copy()

        grp = latest.groupby("brand").agg(
            demand=("demand_index", "mean"),
            sentiment=("sentiment_index", "mean"),
            search=("search_index", "mean"),
            ad=("ad_index", "mean"),
            price=("price_index", "mean"),
        ).reset_index()

        grp["demand_n"] = self._norm_to_100(grp["demand"])
        grp["sent_n"] = self._norm_to_100(grp["sentiment"] * 100.0)
        grp["search_n"] = self._norm_to_100(grp["search"])
        grp["ad_n"] = self._norm_to_100(grp["ad"])
        grp["price_value_n"] = 100.0 - self._norm_to_100(grp["price"])

        grp["score"] = (
            0.35 * grp["demand_n"]
            + 0.25 * grp["sent_n"]
            + 0.20 * grp["search_n"]
            + 0.10 * grp["ad_n"]
            + 0.10 * grp["price_value_n"]
        )

        grp = grp.sort_values("score", ascending=False).head(3)

        return [
            {
                "brand": str(row["brand"]),
                "vals": [
                    round(self._safe_float(row["demand_n"]), 1),
                    round(self._safe_float(row["sent_n"]), 1),
                    round(self._safe_float(row["search_n"]), 1),
                    round(self._safe_float(row["ad_n"]), 1),
                    round(self._safe_float(row["price_value_n"]), 1),
                ],
            }
            for _, row in grp.iterrows()
        ]

    def _build_radar_by_cat(self, master: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for cat in self._top_categories(master, n=3):
            out[cat] = self._build_radar(master, category=cat)
        return out

    def _build_trends_data(self, master: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if master.empty:
            return out

        for cat in self._top_categories(master, n=3):
            sub = master[master["category"].astype(str).str.lower() == cat.lower()].copy()
            if sub.empty:
                continue

            recent = self._latest_by_brand(sub)
            if recent.empty:
                recent = sub.copy()

            top_brands = (
                recent.groupby("brand")["demand_index"]
                .mean()
                .sort_values(ascending=False)
                .head(3)
                .index.astype(str)
                .tolist()
            )

            colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(top_brands))]
            series_list: list[list[float]] = []

            for brand in top_brands:
                bdf = sub[sub["brand"].astype(str) == brand].copy()
                bdf = bdf.sort_values("date")
                daily = (
                    bdf.groupby(bdf["date"].dt.strftime("%Y-%m-%d"))["demand_index"]
                    .mean()
                    .tail(13)
                )
                vals = [round(self._safe_float(v), 2) for v in daily.tolist()]
                series_list.append(vals)

            out[cat] = {
                "brands": top_brands,
                "colors": colors,
                "series": series_list,
            }
        return out

    def _build_competitors(self, master: pd.DataFrame) -> list[dict[str, Any]]:
        if master.empty:
            return []

        latest = self._latest_by_brand(master)
        if latest.empty:
            latest = master.copy()

        entity_grp = latest.groupby(["entity_id", "brand", "product_name", "category"]).agg(
            demand=("demand_index", "mean"),
            price=("price_index", "mean"),
            sent=("sentiment_index", "mean"),
            search=("search_index", "mean"),
            ad=("ad_index", "mean"),
        ).reset_index()

        entity_grp["demand_n"] = self._norm_to_100(entity_grp["demand"])
        entity_grp["sent_n"] = self._norm_to_100(entity_grp["sent"] * 100.0)
        entity_grp["search_n"] = self._norm_to_100(entity_grp["search"])
        entity_grp["ad_n"] = self._norm_to_100(entity_grp["ad"])
        entity_grp["value_n"] = 100.0 - self._norm_to_100(entity_grp["price"])

        entity_grp["mis"] = (
            0.35 * entity_grp["demand_n"]
            + 0.25 * entity_grp["sent_n"]
            + 0.20 * entity_grp["search_n"]
            + 0.10 * entity_grp["ad_n"]
            + 0.10 * entity_grp["value_n"]
        ).round(1)

        entity_grp["score"] = entity_grp["mis"]

        colors = self._brand_color_map(entity_grp["brand"].astype(str).tolist())

        competitors: list[dict[str, Any]] = []
        for _, row in entity_grp.sort_values("score", ascending=False).head(12).iterrows():
            eid = str(row["entity_id"])
            hist = master[master["entity_id"].astype(str) == eid].sort_values("date").copy()

            price_history = (
                hist.groupby(hist["date"].dt.strftime("%Y-%m-%d"))["price_index"]
                .mean()
                .tail(12)
                .tolist()
            )
            sent_history = (
                (hist.groupby(hist["date"].dt.strftime("%Y-%m-%d"))["sentiment_index"]
                 .mean()
                 .tail(12) * 100.0)
                .tolist()
            )

            competitors.append(
                {
                    "entity_id": int(float(eid)) if eid.isdigit() else eid,
                    "name": str(row["brand"]),
                    "product": str(row["product_name"]),
                    "cat": str(row["category"]).replace("_", " ").title(),
                    "color": colors.get(str(row["brand"]), "#C9A84C"),
                    "score": round(self._safe_float(row["score"]), 1),
                    "price": round(self._safe_float(row["price"]), 1),
                    "sent": round(self._safe_float(row["sent"]) * 100.0, 1),
                    "demand": round(self._safe_float(row["demand"]), 1),
                    "mis": round(self._safe_float(row["mis"]), 1),
                    "price_history": [round(self._safe_float(v), 2) for v in price_history],
                    "sent_history": [round(self._safe_float(v), 1) for v in sent_history],
                }
            )

        return competitors

    def _build_events(self, events: pd.DataFrame, category: str) -> list[dict[str, Any]]:
        selected = self._filter_category(events, category)
        if selected.empty:
            return []

        selected = selected.sort_values("date", ascending=False).copy()
        if "brand" not in selected.columns:
            selected["brand"] = ""
        if "category" not in selected.columns:
            selected["category"] = ""
        if "signal_story" not in selected.columns:
            selected["signal_story"] = selected.get("event_description", "")
        if "event_description" not in selected.columns:
            selected["event_description"] = ""

        brand_colors = self._brand_color_map(selected["brand"].astype(str).unique().tolist())

        rows = []
        for _, row in selected.head(6).iterrows():
            rows.append(
                {
                    "color": brand_colors.get(str(row.get("brand", "")), "#C9A84C"),
                    "brand": str(row.get("brand", "")),
                    "cat": str(row.get("category", "")).lower(),
                    "explanation": str(row.get("signal_story", row.get("event_description", ""))),
                    "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
                }
            )
        return rows

    def _build_buzz(self, master: pd.DataFrame, category: str = "all") -> list[float]:
        selected = self._filter_category(master, category)
        if selected.empty:
            return []

        if "buzz_score" in selected.columns and not selected["buzz_score"].isna().all():
            return self._daily_series(selected, "buzz_score", 56)

        temp = selected.copy()
        temp["buzz_proxy"] = (
            pd.to_numeric(temp.get("search_index", 0.0), errors="coerce").fillna(0.0)
            + 8.0 * pd.to_numeric(temp.get("ad_index", 0.0), errors="coerce").fillna(0.0)
        )
        temp["buzz_proxy"] = self._norm_to_100(temp["buzz_proxy"])
        return self._daily_series(temp, "buzz_proxy", 56)

    def _build_category_sparkline(self, master: pd.DataFrame, category: str) -> list[float]:
        selected = self._filter_category(master, category)
        return self._daily_series(selected, "demand_index", 24)

    def _build_regime(self, regime: pd.DataFrame) -> list[dict[str, Any]]:
        if regime.empty:
            return []

        out = []
        for _, row in regime.tail(12).iterrows():
            regime_type = (
                row.get("regime_type")
                or row.get("shift_type")
                or row.get("type")
                or "neutral"
            )
            out.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
                    "regime_type": str(regime_type),
                }
            )
        return out

    def _build_report(self, master: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, Any]:
        if master.empty:
            return {
                "top_cat": None,
                "top_cat_demand": 0.0,
                "top_cat_sent": 0.0,
                "best_brand": None,
                "best_brand_health": 0.0,
                "price_30d_chg": 0.0,
                "headphones_sent": 0.0,
                "total_brands": 0,
                "total_entities": 0,
                "total_alerts": 0,
            }

        latest_date, _ = self._latest_two_dates(master)
        latest = master[master["date"] == latest_date].copy() if latest_date is not None else master.copy()

        cat_grp = latest.groupby("category").agg(
            demand=("demand_index", "mean"),
            sent=("sentiment_index", "mean"),
        ).reset_index()

        if cat_grp.empty:
            top_cat = None
            top_cat_demand = 0.0
            top_cat_sent = 0.0
        else:
            top_row = cat_grp.sort_values("demand", ascending=False).iloc[0]
            top_cat = str(top_row["category"]).replace("_", " ").title()
            top_cat_demand = round(self._safe_float(top_row["demand"]), 1)
            top_cat_sent = round(self._safe_float(top_row["sent"]) * 100.0, 1)

        radar_all = self._build_radar(master, category="all")
        best_brand = radar_all[0]["brand"] if radar_all else None
        best_brand_health = max((vals["vals"][0] for vals in radar_all), default=0.0)

        last_30 = master.sort_values("date").copy()
        max_date = last_30["date"].max()
        min_cut = max_date - pd.Timedelta(days=29)
        last_30 = last_30[last_30["date"] >= min_cut].copy()

        if last_30.empty:
            price_30d_chg = 0.0
        else:
            daily_price = last_30.groupby(last_30["date"].dt.strftime("%Y-%m-%d"))["price_index"].mean()
            first_p = self._safe_float(daily_price.iloc[0])
            last_p = self._safe_float(daily_price.iloc[-1])
            price_30d_chg = self._pct_change(last_p, first_p)

        headphones = self._filter_category(master, "headphones")
        latest_hp = self._latest_by_brand(headphones)
        headphones_sent = round(self._safe_float(latest_hp["sentiment_index"].mean()) * 100.0, 1) if not latest_hp.empty else 0.0

        return {
            "top_cat": top_cat,
            "top_cat_demand": top_cat_demand,
            "top_cat_sent": top_cat_sent,
            "best_brand": best_brand,
            "best_brand_health": round(self._safe_float(best_brand_health), 1),
            "price_30d_chg": price_30d_chg,
            "headphones_sent": headphones_sent,
            "total_brands": int(master["brand"].nunique()) if "brand" in master.columns else 0,
            "total_entities": int(master["entity_id"].nunique()) if "entity_id" in master.columns else 0,
            "total_alerts": int(len(alerts)),
        }