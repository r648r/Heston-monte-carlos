"""Helpers to convert Monte Carlo outputs to Polymarket-like probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .stats_tools import (
    ensure_numpy_array,
    probability_confidence_interval,
    prob_above,
    prob_between,
)


@dataclass(slots=True, frozen=True)
class ProbabilityEstimate:
    probability: float
    ci: tuple[float, float]


def compute_binary_market_probs(
    final_prices: np.ndarray,
    thresholds: Sequence[float],
) -> dict[float, ProbabilityEstimate]:
    """Return {K: P(S_T > K)} plus confidence intervals."""
    prices = ensure_numpy_array(final_prices)
    n_obs = prices.size
    probabilities: dict[float, float] = {}
    for level in thresholds:
        k = float(level)
        prob = prob_above(prices, k)
        probabilities[k] = ProbabilityEstimate(
            probability=prob,
            ci=probability_confidence_interval(prob, n_obs),
        )
    return probabilities


def compute_bucket_market_probs(
    final_prices: np.ndarray,
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], ProbabilityEstimate]:
    """Return {(K1, K2): prob, ci} for each bucket definition."""
    prices = ensure_numpy_array(final_prices)
    n_obs = prices.size
    distribution: dict[tuple[float, float], float] = {}
    for low, high in buckets:
        bucket = (float(low), float(high))
        prob = prob_between(prices, bucket[0], bucket[1])
        distribution[bucket] = ProbabilityEstimate(
            probability=prob,
            ci=probability_confidence_interval(prob, n_obs),
        )
    return distribution


def compute_edges(
    model_probs: Mapping[float, float],
    market_probs: Mapping[float, float],
) -> dict[float, float]:
    """Return the additive edge model_prob - market_prob for comparable keys."""
    edges: dict[float, float] = {}
    for key, model_prob in model_probs.items():
        base_prob = model_prob
        if isinstance(model_prob, ProbabilityEstimate):
            base_prob = model_prob.probability
        market_prob = market_probs.get(key)
        if market_prob is None:
            continue
        edges[key] = float(base_prob - market_prob)
    return edges
