"""Market regime detection and jump parameter heuristics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sharpe_proxy(series: pd.Series) -> float:
    if series is None or series.dropna().empty:
        return 0.0
    clean = series.dropna()
    mean_return = float(clean.mean())
    std_return = float(clean.std(ddof=0))
    if std_return == 0:
        logger.warning("Zero return volatility encountered, applying epsilon to avoid division by zero.")
        std_return = 1e-8
    return (mean_return / std_return) * np.sqrt(365)


def detect_regime(returns: pd.Series, window: int = 60) -> tuple[str, float]:
    """Infer a market regime using a smoothed Sharpe proxy with confirmation."""
    if returns is None or returns.dropna().empty:
        raise ValueError("Cannot detect regime because returns series is empty.")

    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("Not enough returns to evaluate the regime.")

    recent = clean_returns.tail(window)
    if len(recent) < max(10, window // 2):
        raise ValueError("Insufficient data for regime detection.")
    if len(recent) < window:
        logger.warning("Using only %s data points for regime detection (requested %s).", len(recent), window)

    confirmation_available = len(clean_returns) >= 2 * window
    previous = clean_returns.iloc[-2 * window : -window] if confirmation_available else pd.Series(dtype=float)

    current_score = _sharpe_proxy(recent)
    previous_score = _sharpe_proxy(previous) if not previous.empty else None
    smoothed_score = (
        current_score
        if previous_score is None
        else 0.7 * current_score + 0.3 * previous_score
    )

    bullish_threshold = 0.5
    bearish_threshold = -0.5

    bullish = smoothed_score > bullish_threshold
    bearish = smoothed_score < bearish_threshold

    if bullish:
        confirmed = previous_score is not None and previous_score > 0.25
        if confirmed or current_score > bullish_threshold + 0.3:
            return "BULLISH", 1.2
    if bearish:
        confirmed = previous_score is not None and previous_score < -0.25
        if confirmed or current_score < bearish_threshold - 0.3:
            return "BEARISH", 0.8
    return "NEUTRAL", 1.0


def estimate_jump_parameters(returns: pd.Series, threshold_sigma: float = 3.0) -> tuple[float, float, float]:
    """Estimate jump frequency (annualized) and magnitude using a sigma filter."""
    defaults = (0.01 * 365, 0.0, 0.05)
    if returns is None:
        logger.warning("Returns input is None; falling back to default jump parameters.")
        return defaults

    clean_returns = returns.dropna()
    if clean_returns.empty:
        logger.warning("Empty returns series when estimating jumps; using defaults.")
        return defaults

    mean_return = float(clean_returns.mean())
    std_return = float(clean_returns.std(ddof=0))
    if std_return == 0:
        logger.warning("Return volatility is zero; using default jump parameters.")
        return defaults

    jumps = clean_returns[np.abs(clean_returns - mean_return) > threshold_sigma * std_return]

    if jumps.empty:
        logger.info("No significant jumps detected with threshold %.1fσ; using fallback intensity.", threshold_sigma)
        return defaults

    lambda_daily = float(len(jumps) / len(clean_returns))
    lambda_jump = lambda_daily * 365.0
    mu_jump = float(jumps.mean())
    sigma_jump = float(jumps.std(ddof=0) or defaults[2])

    return lambda_jump, mu_jump, sigma_jump
