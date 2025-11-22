"""Statistical helpers for post-processing Monte Carlo simulations."""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def ensure_numpy_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Statistics require a non-empty array of outcomes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Statistics input contains NaN or infinite values.")
    return arr


def probability_confidence_interval(probability: float, sample_size: int, z_score: float = 1.96) -> tuple[float, float]:
    """Return a symmetric normal-approximation CI for a Bernoulli proportion."""
    if sample_size <= 0:
        raise ValueError("Sample size must be positive to compute a confidence interval.")
    variance = probability * (1.0 - probability)
    margin = z_score * math.sqrt(max(variance, 0.0) / sample_size)
    return (
        float(max(0.0, probability - margin)),
        float(min(1.0, probability + margin)),
    )


def calculate_statistics(final_prices: np.ndarray, target_price: float | None = None) -> dict:
    """Return descriptive statistics and guard-rail diagnostics."""
    prices = ensure_numpy_array(final_prices)
    mean_val = float(np.mean(prices))
    std_val = float(np.std(prices))

    stats = {
        "mean": mean_val,
        "median": float(np.median(prices)),
        "std": std_val,
        "min": float(np.min(prices)),
        "max": float(np.max(prices)),
    }

    percentiles = {
        "1": float(np.percentile(prices, 1)),
        "5": float(np.percentile(prices, 5)),
        "10": float(np.percentile(prices, 10)),
        "25": float(np.percentile(prices, 25)),
        "50": float(np.percentile(prices, 50)),
        "75": float(np.percentile(prices, 75)),
        "90": float(np.percentile(prices, 90)),
        "95": float(np.percentile(prices, 95)),
        "99": float(np.percentile(prices, 99)),
    }

    if not np.all(np.diff(list(percentiles.values())) >= 0):
        raise ValueError("Computed percentiles are not ordered; check the input data.")

    stats["percentiles"] = percentiles
    stats["ci_95"] = (
        float(np.percentile(prices, 2.5)),
        float(np.percentile(prices, 97.5)),
    )

    sample_size = len(prices)
    se_mean = std_val / np.sqrt(sample_size)
    stats["se_mean"] = float(se_mean)

    if target_price is not None:
        stats["target_price"] = target_price
        prob = prob_above(prices, target_price)
        stats["prob_above_target"] = prob
        stats["prob_above_ci"] = probability_confidence_interval(prob, sample_size)

    if mean_val != 0 and std_val > abs(mean_val) * 10:
        logger.warning(
            "Distribution appears extremely wide (std/mean=%.1f). Verify parameter inputs.",
            std_val / abs(mean_val),
        )
    if abs(mean_val) > 0 and se_mean > abs(mean_val) * 0.3:
        logger.warning(
            "Monte Carlo standard error (%.4f) exceeds 30%% of the mean %.4f; consider increasing path count.",
            se_mean,
            mean_val,
        )

    return stats


def prob_above(final_prices: np.ndarray, level: float) -> float:
    """Return P(S_T > level)."""
    prices = ensure_numpy_array(final_prices)
    return float(np.mean(prices > level))


def prob_between(final_prices: np.ndarray, low: float, high: float) -> float:
    """Return P(low <= S_T < high)."""
    if low >= high:
        raise ValueError("Bucket lower bound must be strictly less than the upper bound.")
    prices = ensure_numpy_array(final_prices)
    return float(np.mean((prices >= low) & (prices < high)))


def prob_updown(final_prices: np.ndarray, spot: float) -> float:
    """Return the 'Up' probability P(S_T > spot)."""
    return prob_above(final_prices, spot)
