import pandas as pd

from app.config import settings
from app.state_manager import StateManager
from app.utils import normalize_input_df, read_csv_if_exists


class RealtimeIngestor:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def _source_df(self) -> pd.DataFrame:
        source = read_csv_if_exists(settings.QUERY_CSV_PATH)
        if source.empty:
            return pd.DataFrame()

        source = normalize_input_df(source)
        if source.empty:
            return pd.DataFrame()

        source = source.sort_values(["date", "entity_id"]).reset_index(drop=True)
        return source

    def _get_next_available_date(self, source: pd.DataFrame) -> str | None:
        if source.empty:
            return None

        latest_ingested_date = self.state_manager.get_state("last_ingested_date", "")
        all_dates = sorted(source["date"].dropna().unique().tolist())

        for d in all_dates:
            if not latest_ingested_date or d > latest_ingested_date:
                return d

        return None

    def ingest_next_date(self) -> pd.DataFrame:
        source = self._source_df()
        if source.empty:
            return pd.DataFrame()

        next_date = self._get_next_available_date(source)
        if not next_date:
            return pd.DataFrame()

        batch = source[source["date"] == next_date].copy()
        if batch.empty:
            return pd.DataFrame()

        expected_products = 7
        if batch["entity_id"].nunique() != expected_products:
            raise ValueError(
                f"Incomplete daily batch for {next_date}: "
                f"expected {expected_products} products, got {batch['entity_id'].nunique()}"
            )

        batch = batch.sort_values(["date", "entity_id"]).reset_index(drop=True)

        inserted = self.state_manager.insert_actuals(batch)

        if inserted > 0:
            self.state_manager.set_state("last_ingested_date", next_date)
            sim_day = int(self.state_manager.get_state("sim_day", "0")) + 1
            self.state_manager.set_state("sim_day", sim_day)

        return batch

    def ingest_date(self, date: str) -> pd.DataFrame:
        source = self._source_df()
        if source.empty:
            return pd.DataFrame()

        batch = source[source["date"] == date].copy()
        if batch.empty:
            return pd.DataFrame()

        expected_products = 7
        if batch["entity_id"].nunique() != expected_products:
            raise ValueError(
                f"Incomplete daily batch for {date}: "
                f"expected {expected_products} products, got {batch['entity_id'].nunique()}"
            )

        latest_ingested_date = self.state_manager.get_state("last_ingested_date", "")
        if latest_ingested_date and date <= latest_ingested_date:
            return pd.DataFrame()

        batch = batch.sort_values(["date", "entity_id"]).reset_index(drop=True)

        inserted = self.state_manager.insert_actuals(batch)
        if inserted > 0:
            self.state_manager.set_state("last_ingested_date", date)
            sim_day = int(self.state_manager.get_state("sim_day", "0")) + 1
            self.state_manager.set_state("sim_day", sim_day)

        return batch