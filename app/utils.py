import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def bootstrap_seed_inputs() -> None:
    repo_seed_dir = settings.BASE_DIR / "data" / "input"
    runtime_input_dir = settings.INPUT_DIR

    if repo_seed_dir.resolve() == runtime_input_dir.resolve():
        return

    if not repo_seed_dir.exists():
        return

    runtime_input_dir.mkdir(parents=True, exist_ok=True)

    for item in repo_seed_dir.iterdir():
        target = runtime_input_dir / item.name
        if not target.exists() and item.is_file():
            shutil.copy2(item, target)


def ensure_directories() -> None:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_seed_inputs()

    if not settings.REGISTRY_PATH.exists():
        atomic_write_json(
            settings.REGISTRY_PATH,
            {
                "active_version": None,
                "last_stable_version": None,
                "versions": {},
            },
        )


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    text_like_cols = {
        "date": "string",
        "entity_id": "string",
        "brand": "string",
        "product_name": "string",
        "category": "string",
        "event_type": "string",
        "event_title": "string",
        "event_description": "string",
        "impact_direction": "string",
        "signal_story": "string",
        "priority": "string",
        "event_effect_hint": "string",
        "linked_marker_date": "string",
        "linked_marker_type": "string",
        "event_id": "string",
        "event_scope": "string",
        "event_name": "string",
        "signal_target": "string",
        "notes": "string",
    }

    try:
        return pd.read_csv(path, low_memory=False, dtype=text_like_cols)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_csv(df, path)


def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    rename_map = {
        "product_id": "entity_id",
        "sku": "entity_id",
        "asin": "entity_id",
    }
    out = out.rename(columns=rename_map)

    for col in settings.REQUIRED_INPUT_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    defaults = {
        "brand": "",
        "product_name": "",
        "category": "",
        "demand_index": 0.0,
        "price_index": 0.0,
        "sentiment_index": 0.0,
        "search_index": 0.0,
        "ad_index": 0.0,
        "event_type": "organic",
        "event_title": "",
        "event_description": "",
        "impact_direction": "neutral",
        "signal_story": "",
        "priority": "medium",
        "event_effect_hint": "organic_continuation",
    }

    alias_map = {
        "price_new": "price_index",
        "search_interest": "search_index",
        "sentiment_mean": "sentiment_index",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["entity_id"] = out["entity_id"].astype(str)

    numeric_cols = [
        "demand_index",
        "price_index",
        "sentiment_index",
        "search_index",
        "ad_index",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["date", "entity_id"])
    out = out.sort_values(["date", "entity_id"]).reset_index(drop=True)

    final_cols = [
        "date",
        "entity_id",
        "brand",
        "product_name",
        "category",
        "demand_index",
        "price_index",
        "sentiment_index",
        "search_index",
        "ad_index",
        "event_type",
        "event_title",
        "event_description",
        "impact_direction",
        "signal_story",
        "priority",
        "event_effect_hint",
    ]

    return out[final_cols]


def build_training_frame(state_manager) -> pd.DataFrame:
    from app.feature_builder import FeatureBuilder

    fb = FeatureBuilder()
    master_df = fb.load_master_training_source()

    actuals_df = state_manager.read_sql(
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

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "entity_id"]).drop_duplicates(
        ["date", "entity_id"], keep="last"
    )
    return fb.build_training_frame(combined, forecast_horizon=1)