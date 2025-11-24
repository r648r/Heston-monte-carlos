"""Heuristic estimation of Heston model parameters based on market data."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EstimatedHestonParams:
    """Container for the main Heston parameters."""

    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    V0: float


class HestonParameterEstimator:
    """Provide simple heuristics for calibrating the Heston model."""

    def __init__(self, min_sigma_v: float = 0.3, max_sigma_v: float = 1.5) -> None:
        self.min_sigma_v = min_sigma_v
        self.max_sigma_v = max_sigma_v

    def estimate_params(
        self,
        returns: pd.Series,
        vol_annual: float,
        regime_name: str,
        regime_factor: float,
        long_window: int = 180,
        vol_window_for_volofvol: int = 30,
    ) -> EstimatedHestonParams:
        """Estimate parameters from realized data and a detected regime."""
        if returns is None or returns.dropna().empty:
            raise ValueError("Cannot estimate parameters without returns.")
        if vol_annual <= 0 or not np.isfinite(vol_annual):
            raise ValueError("Realized volatility must be a positive finite number.")

        clean_returns = returns.dropna()

        mu_daily = float(clean_returns.mean())
        mu_annual = mu_daily * 365
        mu = float(np.clip(mu_annual * regime_factor, -1.0, 1.0))
        logger.info("Estimated drift %.4f (regime=%s, factor=%.2f).", mu, regime_name, regime_factor)

        V0 = float(vol_annual**2)
        if V0 <= 0:
            raise ValueError("Initial variance V0 must be positive.")

        if len(clean_returns) >= long_window:
            long_sample = clean_returns.tail(long_window)
        else:
            long_sample = clean_returns
            logger.warning(
                "Only %s observations available for long-term variance estimation (requested %s).",
                len(clean_returns),
                long_window,
            )

        theta = float((long_sample**2).mean() * 365)
        if theta <= 0 or not np.isfinite(theta):
            logger.warning("Invalid theta estimate %.4f; falling back to V0.", theta)
            theta = V0

        rolling_vol = clean_returns.rolling(vol_window_for_volofvol).std().dropna() * np.sqrt(365)
        if rolling_vol.dropna().empty:
            logger.warning(
                "Insufficient data to estimate vol-of-vol (window=%s). Falling back to minimum sigma_v.",
                vol_window_for_volofvol,
            )
            sigma_v = self.min_sigma_v
            vol_series_for_corr = pd.Series(dtype=float)
        else:
            vol_of_vol = float(rolling_vol.std(ddof=0))
            if not np.isfinite(vol_of_vol) or vol_of_vol == 0:
                logger.warning("Degenerate vol-of-vol estimate; using minimum sigma_v.")
                sigma_v = self.min_sigma_v
            else:
                sigma_v = float(np.clip(vol_of_vol, self.min_sigma_v, self.max_sigma_v))
            vol_series_for_corr = rolling_vol

        vol_long_term = math.sqrt(theta) if theta > 0 else vol_annual
        if vol_long_term <= 0:
            vol_long_term = vol_annual
        ratio = vol_annual / vol_long_term if vol_long_term > 0 else 1.0
        kappa = 1.5
        if ratio > 1.5:
            kappa = 2.5
        elif ratio < 0.7:
            kappa = 1.0
        logger.info("Estimated kappa %.2f using volatility ratio %.2f.", kappa, ratio)

        rho = -0.5
        if len(vol_series_for_corr) > 5:
            vol_changes = vol_series_for_corr.diff().dropna()
            if len(vol_changes) > 5:
                aligned_returns = clean_returns.reindex(vol_changes.index).dropna()
                min_len = min(len(aligned_returns), len(vol_changes))
                if min_len > 5:
                    aligned_returns = aligned_returns.tail(min_len)
                    vol_changes = vol_changes.tail(min_len)
                    corr = aligned_returns.corr(vol_changes)
                    if np.isfinite(corr):
                        rho = float(np.clip(corr, -0.9, -0.1))
                        logger.info("Empirical rho=-%.2f from return/volatility correlation.", abs(rho))
                    else:
                        logger.warning("Correlation computation returned NaN; using default rho %.2f.", rho)
        else:
            logger.warning("Not enough volatility data to estimate rho; using default %.2f.", rho)

        if not (-1.0 <= rho <= 0.0):
            rho = float(np.clip(rho, -0.9, -0.1))

        theta_safe = max(theta, 1e-8)
        required_kappa = (sigma_v**2 + 1e-10) / (2 * theta_safe)
        if required_kappa > kappa and np.isfinite(required_kappa):
            logger.warning(
                "Increasing kappa from %.3f to %.3f to satisfy the Feller condition.",
                kappa,
                required_kappa,
            )
            kappa = required_kappa

        feller_gap = 2 * kappa * theta_safe - sigma_v**2
        if feller_gap <= 0:
            sigma_cap = math.sqrt(max(2 * kappa * theta_safe * 0.999, 1e-8))
            if sigma_cap < sigma_v:
                logger.warning(
                    "Clipping sigma_v from %.3f to %.3f to maintain positive variance dynamics.",
                    sigma_v,
                    sigma_cap,
                )
                sigma_v = sigma_cap

        return EstimatedHestonParams(mu=mu, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, V0=V0)
