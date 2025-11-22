"""Implementation of the Heston model with Merton-style jump diffusion."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimulationResult:
    """Container for simulated price and variance paths."""

    prices: np.ndarray
    variances: np.ndarray


class HestonJumpDiffusionModel:
    """Monte Carlo simulator for the Heston stochastic volatility model with jumps."""

    def __init__(
        self,
        S0: float,
        V0: float,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        lambda_jump: float,
        mu_jump: float,
        sigma_jump: float,
    ) -> None:
        self.S0 = float(S0)
        self.V0 = float(V0)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.lambda_jump = float(lambda_jump)
        self.mu_jump = float(mu_jump)
        self.sigma_jump = float(sigma_jump)
        self._jump_drift = self._compute_jump_drift_adjustment()
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Ensure parameters obey the guard-rail constraints."""
        if self.V0 <= 0 or not np.isfinite(self.V0):
            raise ValueError("Initial variance V0 must be positive.")
        if self.theta <= 0:
            raise ValueError("Long-term variance theta must be positive.")
        if self.sigma_v <= 0:
            raise ValueError("Vol-of-vol sigma_v must be positive.")
        if self.kappa <= 0:
            raise ValueError("Mean-reversion speed kappa must be positive.")
        if not (-1 <= self.rho <= 1):
            raise ValueError("Correlation rho must be within [-1, 1].")
        if self.lambda_jump < 0:
            raise ValueError("Jump intensity lambda_jump must be non-negative.")
        feller_term = 2 * self.kappa * self.theta
        if feller_term <= self.sigma_v**2:
            logger.warning(
                "Parameters violate the Feller condition (2κθ=%.4f <= σ_v²=%.4f).",
                feller_term,
                self.sigma_v**2,
            )

    def _compute_jump_drift_adjustment(self) -> float:
        if self.lambda_jump <= 0:
            return 0.0
        jump_expectation = math.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0
        return self.lambda_jump * jump_expectation

    def simulate_paths(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator | None = None,
        full_paths: bool = True,
    ) -> SimulationResult:
        """Simulate joint price and variance paths."""
        if T <= 0 or n_steps <= 0 or n_paths <= 0:
            raise ValueError("Simulation horizon, steps, and paths must be positive.")

        rng = rng or np.random.default_rng()
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((n_paths, n_steps + 1), dtype=float)
        V = np.zeros((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        rho_term = np.sqrt(max(1.0 - self.rho**2, 0.0))

        for step in range(n_steps):
            v_prev = np.maximum(V[:, step], 0.0)
            sqrt_v_prev = np.sqrt(v_prev)

            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + rho_term * z2

            if self.lambda_jump > 0:
                poisson_rates = rng.poisson(self.lambda_jump * dt, size=n_paths)
                jump_component = np.zeros(n_paths)
                mask = poisson_rates > 0
                if np.any(mask):
                    jump_component[mask] = rng.normal(
                        loc=self.mu_jump * poisson_rates[mask],
                        scale=self.sigma_jump * np.sqrt(poisson_rates[mask]),
                        size=mask.sum(),
                    )
            else:
                jump_component = np.zeros(n_paths)

            V[:, step + 1] = (
                V[:, step]
                + self.kappa * (self.theta - v_prev) * dt
                + self.sigma_v * sqrt_v_prev * sqrt_dt * w2
            )
            V[:, step + 1] = np.maximum(V[:, step + 1], 0.0)

            drift = (self.mu - self._jump_drift - 0.5 * v_prev) * dt
            diffusion = sqrt_v_prev * sqrt_dt * w1
            S[:, step + 1] = S[:, step] * np.exp(drift + diffusion + jump_component)

        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(V)):
            raise ValueError("Simulation produced non-finite values; check parameters.")

        if not full_paths:
            S = S[:, -1]
            V = V[:, -1]

        return SimulationResult(prices=S, variances=V)
