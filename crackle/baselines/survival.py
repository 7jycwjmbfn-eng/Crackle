from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackle.data.common import sigmoid


@dataclass
class LogisticHazard:
    model: str
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    selected_feature_indices: list[int] | None = None

    def logits(self, features: np.ndarray) -> np.ndarray:
        x_in = np.asarray(features, dtype=np.float64)
        if self.selected_feature_indices is not None and x_in.shape[1] != self.mean.size:
            x_in = x_in[:, self.selected_feature_indices]
        x = (x_in - self.mean[None, :]) / self.scale[None, :]
        x[:, 0] = 1.0
        return x @ self.weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(features))

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "crackle_model_v1",
            "kind": "logistic_hazard",
            "model": self.model,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "selected_feature_indices": self.selected_feature_indices,
            "uses_future_labels": False,
            "oracle_row": False,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "LogisticHazard":
        return cls(
            model=str(payload["model"]),
            feature_names=[str(item) for item in payload["feature_names"]],
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            selected_feature_indices=payload.get("selected_feature_indices"),
        )


def fit_logistic_hazard(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model: str,
    feature_names: list[str],
    selected_feature_indices: list[int] | None = None,
    seed: int = 20260605,
    epochs: int = 220,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> LogisticHazard:
    x_raw = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x_raw.ndim != 2:
        raise ValueError(f"features must be 2D, got {x_raw.shape}")
    if y.size != x_raw.shape[0]:
        raise ValueError(f"labels have {y.size} entries, expected {x_raw.shape[0]}")
    if np.count_nonzero(y > 0.5) == 0 or np.count_nonzero(y <= 0.5) == 0:
        raise ValueError("logistic hazard training needs both positive events and censored/negative risk bonds")
    mean = np.mean(x_raw, axis=0)
    scale = np.std(x_raw, axis=0)
    mean[0] = 0.0
    scale = np.maximum(scale, 1e-6)
    scale[0] = 1.0
    x = (x_raw - mean[None, :]) / scale[None, :]
    x[:, 0] = 1.0
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.02, size=(x.shape[1],))
    pos = max(float(np.count_nonzero(y > 0.5)), 1.0)
    neg = max(float(np.count_nonzero(y <= 0.5)), 1.0)
    sample_w = np.where(y > 0.5, 0.5 / pos, 0.5 / neg) * y.size
    beta1, beta2 = 0.9, 0.999
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    for epoch in range(1, int(epochs) + 1):
        logits = x @ weights
        prob = sigmoid(logits)
        grad = (x.T @ ((prob - y) * sample_w)) / y.size + float(l2) * weights
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**epoch)
        v_hat = v / (1.0 - beta2**epoch)
        weights -= float(lr) * m_hat / (np.sqrt(v_hat) + 1e-8)
    return LogisticHazard(
        model=model,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights,
        selected_feature_indices=selected_feature_indices,
    )


def fit_logistic_hazard_with_pairwise(
    features: np.ndarray,
    labels: np.ndarray,
    pairwise_diffs: np.ndarray | None,
    *,
    model: str,
    feature_names: list[str],
    selected_feature_indices: list[int] | None = None,
    seed: int = 20260605,
    epochs: int = 220,
    lr: float = 0.08,
    l2: float = 1e-4,
    rank_weight: float = 0.35,
    focal_gamma: float = 0.0,
) -> LogisticHazard:
    x_raw = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x_raw.ndim != 2:
        raise ValueError(f"features must be 2D, got {x_raw.shape}")
    if y.size != x_raw.shape[0]:
        raise ValueError(f"labels have {y.size} entries, expected {x_raw.shape[0]}")
    if np.count_nonzero(y > 0.5) == 0 or np.count_nonzero(y <= 0.5) == 0:
        raise ValueError("logistic hazard training needs both positive events and censored/negative risk bonds")
    mean = np.mean(x_raw, axis=0)
    scale = np.std(x_raw, axis=0)
    mean[0] = 0.0
    scale = np.maximum(scale, 1e-6)
    scale[0] = 1.0
    x = (x_raw - mean[None, :]) / scale[None, :]
    x[:, 0] = 1.0
    diffs = None
    if pairwise_diffs is not None:
        raw_diffs = np.asarray(pairwise_diffs, dtype=np.float64)
        if raw_diffs.ndim != 2 or raw_diffs.shape[1] != x_raw.shape[1]:
            raise ValueError(f"pairwise_diffs shape {raw_diffs.shape} is incompatible with features {x_raw.shape}")
        if raw_diffs.shape[0] > 0:
            diffs = raw_diffs / scale[None, :]
            diffs[:, 0] = 0.0
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.02, size=(x.shape[1],))
    pos = max(float(np.count_nonzero(y > 0.5)), 1.0)
    neg = max(float(np.count_nonzero(y <= 0.5)), 1.0)
    sample_w = np.where(y > 0.5, 0.5 / pos, 0.5 / neg) * y.size
    beta1, beta2 = 0.9, 0.999
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    for epoch in range(1, int(epochs) + 1):
        logits = x @ weights
        prob = sigmoid(logits)
        residual = prob - y
        if focal_gamma > 0.0:
            pt = np.where(y > 0.5, prob, 1.0 - prob)
            residual = residual * np.power(np.clip(1.0 - pt, 0.0, 1.0), float(focal_gamma))
        grad = (x.T @ (residual * sample_w)) / y.size
        if diffs is not None and diffs.shape[0] > 0 and rank_weight > 0.0:
            rank_prob = sigmoid(diffs @ weights)
            # diff = positive feature - negative feature, so target is 1.
            rank_grad = (diffs.T @ (rank_prob - 1.0)) / diffs.shape[0]
            grad += float(rank_weight) * rank_grad
        grad += float(l2) * weights
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**epoch)
        v_hat = v / (1.0 - beta2**epoch)
        weights -= float(lr) * m_hat / (np.sqrt(v_hat) + 1e-8)
    return LogisticHazard(
        model=model,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights,
        selected_feature_indices=selected_feature_indices,
    )
