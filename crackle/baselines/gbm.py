from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GBMHazard:
    model: str
    backend: str
    feature_names: list[str]
    selected_feature_indices: list[int]
    model_path: str
    model_object: Any | None = None

    def _load_model(self) -> Any:
        if self.model_object is not None:
            return self.model_object
        import joblib

        self.model_object = joblib.load(self.model_path)
        # Crackle active-set inference scores small candidate batches per step.
        # XGBoost's CUDA predictor is slower here unless the candidate matrix
        # also lives on GPU, so default to CPU prediction and keep CUDA as a
        # training acceleration option.
        try:
            self.model_object.set_params(device="cpu")
            self.model_object.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass
        return self.model_object

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x_in = np.asarray(features, dtype=np.float32)
        if x_in.shape[1] != len(self.selected_feature_indices):
            x_in = x_in[:, self.selected_feature_indices]
        model = self._load_model()
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_in)[:, 1]
        elif hasattr(model, "get_booster"):
            prob = model.get_booster().inplace_predict(x_in)
        else:
            raw = np.asarray(model.predict(x_in), dtype=np.float64).reshape(-1)
            prob = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "crackle_model_v1_3",
            "kind": "gbm_hazard",
            "model": self.model,
            "backend": self.backend,
            "feature_names": self.feature_names,
            "selected_feature_indices": self.selected_feature_indices,
            "model_artifact": Path(self.model_path).name,
            "uses_future_labels": False,
            "oracle_row": False,
            "paris_prior_features": False,
            "note": "GBM is trained as a causal discrete-time bond hazard ranker for the Crackle riskset.",
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GBMHazard":
        model_dir = Path(str(payload["model_dir"]))
        artifact = str(payload.get("model_artifact") or "gbm_model.joblib")
        return cls(
            model=str(payload["model"]),
            backend=str(payload.get("backend") or "unknown"),
            feature_names=[str(item) for item in payload.get("feature_names", [])],
            selected_feature_indices=[int(item) for item in payload.get("selected_feature_indices", [])],
            model_path=str(model_dir / artifact),
        )


def fit_gbm_hazard(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model_name: str,
    feature_names: list[str],
    selected_feature_indices: list[int],
    out_dir: Path,
    seed: int,
    n_estimators: int = 320,
    max_depth: int = 5,
    learning_rate: float = 0.045,
    subsample: float = 0.90,
    colsample_bytree: float = 0.90,
    n_jobs: int = 0,
    device: str = "cpu",
) -> GBMHazard:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got {x.shape}")
    if y.size != x.shape[0]:
        raise ValueError(f"labels have {y.size} entries, expected {x.shape[0]}")
    if np.count_nonzero(y == 1) == 0 or np.count_nonzero(y == 0) == 0:
        raise ValueError("GBM hazard training needs both positive and negative rows")

    from xgboost import XGBClassifier

    pos = max(float(np.count_nonzero(y == 1)), 1.0)
    neg = max(float(np.count_nonzero(y == 0)), 1.0)
    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=2.0,
        reg_lambda=2.0,
        scale_pos_weight=float(neg / pos),
        random_state=int(seed),
        n_jobs=int(n_jobs),
        tree_method="hist",
    )
    backend = "xgboost_binary_discrete_hazard_cpu"
    requested_device = str(device or "cpu").lower()
    if requested_device.startswith("cuda") or requested_device == "gpu":
        base_kwargs["device"] = "cuda"
        backend = "xgboost_binary_discrete_hazard_cuda"
    else:
        base_kwargs["device"] = "cpu"

    estimator = XGBClassifier(**base_kwargs)
    try:
        estimator.fit(x, y)
    except Exception:
        if base_kwargs.get("device") != "cuda":
            raise
        base_kwargs["device"] = "cpu"
        backend = "xgboost_binary_discrete_hazard_cpu_fallback"
        estimator = XGBClassifier(**base_kwargs)
        estimator.fit(x, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "gbm_model.joblib"
    import joblib

    joblib.dump(estimator, artifact)
    return GBMHazard(
        model=model_name,
        backend=backend,
        feature_names=feature_names,
        selected_feature_indices=selected_feature_indices,
        model_path=str(artifact),
        model_object=estimator,
    )
