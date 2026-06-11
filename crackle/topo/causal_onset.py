"""Causal (online) onset detectors + lead/false-alarm evaluation (Phase 2.1).

Replaces the Phase-0 retrospective onset detector (max-normalized, future
access) with two strictly causal detectors. At step t both see only
x[0..t]; standardization uses the trailing window x[t-window..t-1].

- rolling z : alarm when (x[t] - mean_past) / std_past > threshold
- CUSUM     : s_t = max(0, s_{t-1} + z_t - k); alarm when s_t > h
              (one-sided, detects sustained upward shift; k = drift)

Evaluation conventions (these are EVALUATION-time quantities; they may be
retrospective without contaminating the detectors):

- instability_step t*    : argmax of the total-damage increment
- growth_start           : first step with cumulative damage >= frac of
                           final damage (default 5%) — "sustained damage
                           growth" reference
- first alarm < growth_start  -> false alarm for that case
- lead = t* - first_alarm, reported only for non-false alarms
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STD_FLOOR_REL = 1e-3
_STD_FLOOR_ABS = 1e-9


def _past_stats(x: np.ndarray, t: int, window: int) -> tuple[float, float]:
    past = x[max(t - window, 0) : t]
    mean = float(past.mean())
    std = float(past.std())
    return mean, max(std, _STD_FLOOR_ABS + _STD_FLOOR_REL * abs(mean))


def rolling_z_alarm(
    x: np.ndarray, *, window: int = 10, threshold: float = 3.0,
    min_history: int = 5,
) -> int | None:
    """First step whose z-score against the trailing window exceeds threshold."""
    x = np.asarray(x, dtype=np.float64)
    for t in range(min_history, x.size):
        mean, std = _past_stats(x, t, window)
        if (x[t] - mean) / std > threshold:
            return t
    return None


def cusum_alarm(
    x: np.ndarray, *, window: int = 10, k: float = 0.5, h: float = 5.0,
    min_history: int = 5,
) -> int | None:
    """One-sided CUSUM on causally standardized residuals."""
    x = np.asarray(x, dtype=np.float64)
    s = 0.0
    for t in range(min_history, x.size):
        mean, std = _past_stats(x, t, window)
        s = max(0.0, s + (x[t] - mean) / std - k)
        if s > h:
            return t
    return None


def growth_start_step(total_damage: np.ndarray, *, frac: float = 0.05) -> int:
    d = np.asarray(total_damage, dtype=np.float64)
    d = d - d[0]
    final = float(d[-1])
    if final <= 0.0:
        return d.size  # never grows: any alarm is false
    hits = np.flatnonzero(d >= frac * final)
    return int(hits[0]) if hits.size else d.size


@dataclass
class CaseEval:
    case_id: str
    signal: str
    detector: str
    threshold: float
    first_alarm: int | None
    t_star: int
    growth_start: int

    @property
    def is_false_alarm(self) -> bool:
        return self.first_alarm is not None and self.first_alarm < self.growth_start

    @property
    def lead(self) -> int | None:
        if self.first_alarm is None or self.is_false_alarm:
            return None
        return self.t_star - self.first_alarm


def evaluate_case(
    case_id: str,
    signal_name: str,
    signal: np.ndarray,
    total_damage: np.ndarray,
    *,
    detector: str,
    threshold: float,
    window: int = 10,
    growth_frac: float = 0.05,
) -> CaseEval:
    from crackle.topo.instability import instability_step

    if detector == "z":
        alarm = rolling_z_alarm(signal, window=window, threshold=threshold)
    elif detector == "cusum":
        alarm = cusum_alarm(signal, window=window, h=threshold)
    else:
        raise ValueError(f"unknown detector {detector!r}")
    return CaseEval(
        case_id=case_id,
        signal=signal_name,
        detector=detector,
        threshold=threshold,
        first_alarm=alarm,
        t_star=instability_step(total_damage),
        growth_start=growth_start_step(total_damage, frac=growth_frac),
    )
