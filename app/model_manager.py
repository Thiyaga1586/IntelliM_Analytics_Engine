import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.config import settings
from app.feature_builder import FeatureBuilder
from app.state_manager import StateManager
from app.utils import atomic_write_json, load_json


class ModelManager:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.models_dir = settings.MODELS_DIR
        self.registry_path = settings.REGISTRY_PATH
        self.feature_builder = FeatureBuilder()
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            atomic_write_json(
                self.registry_path,
                {
                    "active_version": None,
                    "last_stable_version": None,
                    "versions": {},
                },
            )

    def _registry(self) -> dict[str, Any]:
        registry = load_json(
            self.registry_path,
            {
                "active_version": None,
                "last_stable_version": None,
                "versions": {},
            },
        )

        if not isinstance(registry, dict):
            registry = {}

        registry.setdefault("active_version", None)
        registry.setdefault("last_stable_version", None)
        registry.setdefault("versions", {})
        return registry

    def _save_registry(self, registry: dict[str, Any]) -> None:
        registry.setdefault("active_version", None)
        registry.setdefault("last_stable_version", None)
        registry.setdefault("versions", {})
        atomic_write_json(self.registry_path, registry)

    def _next_version(self) -> str:
        return f"xgb_demand_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def _artifact_root_for_version(self, version: str) -> Path | None:
        registry = self._registry()
        entry = registry.get("versions", {}).get(version)
        if not entry:
            return None

        artifact_path = entry.get("artifact_path")
        if not artifact_path:
            return None

        root = Path(artifact_path)
        if not root.exists():
            return None

        return root

    def _resolve_feature_columns(
        self,
        root: Path | None = None,
        registry_entry: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        feature_columns: list[str] = []

        if root is not None:
            feature_columns = load_json(root / "feature_columns.json", [])
            if isinstance(feature_columns, list) and len(feature_columns) > 0:
                return feature_columns

        if registry_entry:
            feature_columns = registry_entry.get("feature_columns", [])
            if isinstance(feature_columns, list) and len(feature_columns) > 0:
                return feature_columns

        if metadata:
            feature_columns = metadata.get("feature_columns", [])
            if isinstance(feature_columns, list) and len(feature_columns) > 0:
                return feature_columns

        return self.feature_builder.get_demand_feature_columns()

    def get_active_version(self) -> str | None:
        registry = self._registry()
        version = registry.get("active_version")

        if not version:
            return None

        root = self._artifact_root_for_version(version)
        if root is None:
            return None

        return version

    def get_active_model_info(self) -> dict[str, Any]:
        registry = self._registry()
        active_version = registry.get("active_version")

        if not active_version:
            return {"active_version": None}

        entry = registry.get("versions", {}).get(active_version, {})
        artifact_path = entry.get("artifact_path")
        root = Path(artifact_path) if artifact_path else None

        metadata = {}
        feature_columns = []

        if root and root.exists():
            metadata = load_json(root / "metadata.json", {})
            feature_columns = self._resolve_feature_columns(root=root, registry_entry=entry, metadata=metadata)
        else:
            metadata = {}
            feature_columns = self._resolve_feature_columns(root=None, registry_entry=entry, metadata={})

        return {
            "active_version": active_version,
            "last_stable_version": registry.get("last_stable_version"),
            "registry_entry": entry,
            "metadata": metadata,
            "feature_columns": feature_columns,
            "artifact_exists": bool(root and root.exists()),
        }

    def get_active_model_bundle(self) -> tuple[xgb.XGBRegressor, list[str], dict[str, Any]]:
        registry = self._registry()
        version = registry.get("active_version")

        if not version:
            raise RuntimeError("No active model found")

        entry = registry.get("versions", {}).get(version, {})
        artifact_path = entry.get("artifact_path")
        if not artifact_path:
            raise RuntimeError(f"Active model {version} has no artifact_path")

        root = Path(artifact_path)
        if not root.exists():
            raise RuntimeError(f"Active model artifact folder missing for version {version}")

        model_path = root / "model_demand.json"
        if not model_path.exists():
            raise RuntimeError(f"Model file missing for active version {version}")

        metadata = load_json(root / "metadata.json", {})
        feature_columns = self._resolve_feature_columns(
            root=root,
            registry_entry=entry,
            metadata=metadata,
        )

        if not feature_columns:
            raise RuntimeError(f"No feature columns available for active version {version}")

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))

        return model, feature_columns, metadata

    def predict_batch(self, infer_df: pd.DataFrame) -> pd.Series:
        if infer_df.empty:
            return pd.Series(dtype=float)

        model, feature_columns, _ = self.get_active_model_bundle()

        if not feature_columns:
            raise RuntimeError("Active model has no feature columns configured")

        X = infer_df.reindex(columns=feature_columns).copy()

        for col in feature_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.fillna(0.0)

        preds = model.predict(X)
        return pd.Series(preds, index=infer_df.index)

    def train_validate_save(
        self,
        train_df: pd.DataFrame,
        target_col: str = None,
        feature_columns: Optional[List[str]] = None,
        min_rows: int = settings.MIN_TRAIN_ROWS,
        trigger_reason: str = "manual",
    ) -> Dict[str, Any]:
        feature_columns = feature_columns or self.feature_builder.get_demand_feature_columns()
        target_col = target_col or self.feature_builder.get_demand_target()

        if train_df.empty:
            return {"accepted": False, "reason": "empty_training_data"}

        df = self.feature_builder.build_training_frame(
            historical_df=train_df,
            feature_columns=feature_columns,
            target_col=target_col,
            forecast_horizon=1,
        )

        if df.empty:
            return {"accepted": False, "reason": "empty_feature_frame"}

        if len(df) < min_rows:
            return {"accepted": False, "reason": f"insufficient_rows:{len(df)}"}

        unique_dates = sorted(df["date"].dropna().unique().tolist())
        if len(unique_dates) < 20:
            return {"accepted": False, "reason": "insufficient_unique_dates"}

        split_idx = int(len(unique_dates) * 0.80)
        if split_idx <= 0 or split_idx >= len(unique_dates):
            return {"accepted": False, "reason": "invalid_train_valid_split"}

        train_dates = unique_dates[:split_idx]
        valid_dates = unique_dates[split_idx:]

        train_part = df[df["date"].isin(train_dates)].copy()
        valid_part = df[df["date"].isin(valid_dates)].copy()

        if train_part.empty:
            return {"accepted": False, "reason": "empty_train_partition"}

        if valid_part.empty or len(valid_part) < 30:
            return {"accepted": False, "reason": "insufficient_validation_rows"}

        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return {"accepted": False, "reason": f"missing_feature_columns:{missing_cols[:5]}"}

        if target_col not in df.columns:
            return {"accepted": False, "reason": f"missing_target:{target_col}"}

        X_train = train_part[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_train = pd.to_numeric(train_part[target_col], errors="coerce")

        X_valid = valid_part[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(valid_part[target_col], errors="coerce")

        valid_mask_train = y_train.notna()
        valid_mask_valid = y_valid.notna()

        X_train = X_train.loc[valid_mask_train]
        y_train = y_train.loc[valid_mask_train]

        X_valid = X_valid.loc[valid_mask_valid]
        y_valid = y_valid.loc[valid_mask_valid]

        if X_train.empty or y_train.empty:
            return {"accepted": False, "reason": "empty_clean_train_partition"}

        if X_valid.empty or y_valid.empty or len(X_valid) < 30:
            return {"accepted": False, "reason": "empty_clean_valid_partition"}

        model = xgb.XGBRegressor(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        valid_pred = model.predict(X_valid)
        mae = float(mean_absolute_error(y_valid, valid_pred))
        rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))

        version = self._next_version()
        model_root = self.models_dir / version
        model_root.mkdir(parents=True, exist_ok=True)

        model.save_model(str(model_root / "model_demand.json"))

        baseline_stats = {
            col: {
                "min": float(X_train[col].min()),
                "q10": float(X_train[col].quantile(0.10)),
                "q50": float(X_train[col].quantile(0.50)),
                "q90": float(X_train[col].quantile(0.90)),
                "max": float(X_train[col].max()),
            }
            for col in feature_columns
        }

        metadata = {
            "version": version,
            "target": target_col,
            "feature_columns": feature_columns,
            "created_at": datetime.utcnow().isoformat(),
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "train_start_date": str(min(train_dates)),
            "train_end_date": str(max(train_dates)),
            "valid_start_date": str(min(valid_dates)),
            "valid_end_date": str(max(valid_dates)),
            "metrics": {"mae": mae, "rmse": rmse},
            "forecast_horizon": 1,
            "model_family": "xgboost_demand",
        }

        atomic_write_json(model_root / "feature_columns.json", feature_columns)
        atomic_write_json(model_root / "baseline_stats.json", baseline_stats)
        atomic_write_json(model_root / "metadata.json", metadata)

        accepted = self._accept_candidate(mae)
        registry = self._registry()
        previous_active = registry.get("active_version")

        registry["versions"][version] = {
            "status": "active" if accepted else "archived",
            "artifact_path": str(model_root),
            "metrics": metadata["metrics"],
            "created_at": metadata["created_at"],
            "feature_columns": feature_columns,
        }

        if accepted:
            if previous_active and previous_active in registry["versions"]:
                registry["versions"][previous_active]["status"] = "archived"
                registry["last_stable_version"] = previous_active
            else:
                registry["last_stable_version"] = version

            registry["active_version"] = version

            self.state_manager.set_state("active_model_version", str(version))
            self.state_manager.set_state("active_baseline_mae", str(mae))
            self.state_manager.set_state(
                "last_retrain_sim_day",
                str(self.state_manager.get_state("sim_day", "0")),
            )
            self.state_manager.set_state("rows_since_last_retrain", "0")
            self.state_manager.set_state("consecutive_perf_drift", "0")
            self.state_manager.set_state("consecutive_rollback_drift", "0")

        self._save_registry(registry)

        self.state_manager.insert_retrain_job(
            {
                "trigger_reason": trigger_reason,
                "old_version": previous_active,
                "new_version": version,
                "status": "accepted" if accepted else "rejected",
                "details_json": json.dumps({"mae": mae, "rmse": rmse}),
            }
        )

        return {
            "accepted": accepted,
            "version": version,
            "mae": mae,
            "rmse": rmse,
            "reason": "activated" if accepted else "candidate_rejected",
        }

    def _accept_candidate(self, challenger_mae: float) -> bool:
        info = self.get_active_model_info()
        if not info.get("active_version"):
            return True

        active_meta = info.get("metadata", {})
        active_mae = float(active_meta.get("metrics", {}).get("mae", 999999.0))
        return challenger_mae <= active_mae * 0.98

    def rollback_to_last_stable(self) -> dict[str, Any]:
        registry = self._registry()
        current_active = registry.get("active_version")
        stable = registry.get("last_stable_version")

        if not stable:
            return {"rolled_back": False, "reason": "no_stable_version"}

        if stable == current_active:
            return {"rolled_back": False, "reason": "already_on_stable"}

        stable_root = self._artifact_root_for_version(stable)
        if stable_root is None:
            return {"rolled_back": False, "reason": "stable_artifact_missing"}

        if current_active and current_active in registry["versions"]:
            registry["versions"][current_active]["status"] = "rolled_back"

        registry["active_version"] = stable
        registry["versions"][stable]["status"] = "active"
        self._save_registry(registry)

        stable_meta = load_json(stable_root / "metadata.json", {})
        stable_mae = stable_meta.get("metrics", {}).get("mae", 999999.0)

        self.state_manager.set_state("active_model_version", str(stable))
        self.state_manager.set_state("active_baseline_mae", str(stable_mae))
        self.state_manager.set_state("consecutive_rollback_drift", "0")

        self.state_manager.insert_retrain_job(
            {
                "trigger_reason": "rollback",
                "old_version": current_active,
                "new_version": stable,
                "status": "rolled_back",
                "details_json": json.dumps({}),
            }
        )

        return {"rolled_back": True, "active_version": stable}