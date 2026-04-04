import pandas as pd

from app.config import settings
from app.utils import read_csv_if_exists


class EventAttributor:
    def __init__(self):
        self.events_path = settings.EVENTS_CSV_PATH

    def _events_df(self) -> pd.DataFrame:
        df = read_csv_if_exists(self.events_path)
        if df.empty:
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df

    def find_likely_event(
        self,
        entity_id: str,
        target_date: str,
        signal_target: str | None = None,
        lookback_days: int = 5,
    ) -> dict:
        df = self._events_df()
        if df.empty:
            return {}

        target_dt = pd.to_datetime(target_date, errors="coerce")
        if pd.isna(target_dt):
            return {}

        start_dt = target_dt - pd.Timedelta(days=lookback_days)

        candidates = df[
            (df["date"] >= start_dt) &
            (df["date"] <= target_dt)
        ].copy()

        if candidates.empty:
            return {}

        if "entity_id" in candidates.columns:
            candidates = candidates[
                (candidates["entity_id"].astype(str) == str(entity_id)) |
                (candidates["entity_id"].astype(str) == "ALL")
            ]

        if candidates.empty:
            return {}

        if signal_target and "signal_target" in candidates.columns:
            preferred = candidates[candidates["signal_target"].astype(str) == str(signal_target)].copy()
            if not preferred.empty:
                candidates = preferred

        if "event_strength" not in candidates.columns:
            candidates["event_strength"] = 0.5

        candidates["event_strength"] = pd.to_numeric(candidates["event_strength"], errors="coerce").fillna(0.5)

        # strongest and latest event wins
        candidates = candidates.sort_values(["event_strength", "date"], ascending=[False, False])

        best = candidates.iloc[0]

        return {
            "event_id": best.get("event_id"),
            "event_date": str(best.get("date").date()) if pd.notna(best.get("date")) else None,
            "event_type": best.get("event_type"),
            "event_title": best.get("event_title", best.get("event_name")),
            "event_description": best.get("event_description", best.get("notes")),
            "event_scope": best.get("event_scope"),
            "signal_target": best.get("signal_target"),
            "event_strength": float(best.get("event_strength", 0.5)),
            "impact_direction": best.get("impact_direction"),
            "signal_story": best.get("signal_story"),
            "priority": best.get("priority"),
            "notes": best.get("notes"),
        }