"""Light-touch sanity checks for the heston_crypto_sim package."""

from __future__ import annotations

import numpy as np
import pandas as pd

from heston_crypto_sim.models.heston_jump import HestonJumpDiffusionModel
from heston_crypto_sim.params_estimator import HestonParameterEstimator
from heston_crypto_sim.stats_tools import calculate_statistics
from heston_crypto_sim.time_scale_analysis import run_time_scale_sweep


def test_parameter_estimation() -> None:
    returns = pd.Series(np.linspace(-0.02, 0.02, 200))
    estimator = HestonParameterEstimator()
    params = estimator.estimate_params(
        returns=returns,
        vol_annual=0.6,
        regime_name="NEUTRAL",
        regime_factor=1.0,
    )
    assert params.V0 > 0 and params.theta > 0
    assert -1.0 <= params.rho <= -0.1


def test_model_simulation_shapes() -> None:
    model = HestonJumpDiffusionModel(
        S0=30_000,
        V0=0.4**2,
        mu=0.1,
        kappa=1.5,
        theta=0.4**2,
        sigma_v=0.6,
        rho=-0.5,
        lambda_jump=0.01,
        mu_jump=0.0,
        sigma_jump=0.05,
    )
    result = model.simulate_paths(T=5 / 365, n_steps=5, n_paths=128)
    assert result.prices.shape == (128, 6)
    assert result.variances.shape == (128, 6)


def test_statistics_output() -> None:
    sample = np.linspace(20_000, 40_000, 100)
    stats = calculate_statistics(sample, target_price=30_000)
    assert stats["prob_above_target"] == 0.5
    assert stats["percentiles"]["50"] == 30_000.0


def test_time_scale_sweep() -> None:
    model = HestonJumpDiffusionModel(
        S0=30_000,
        V0=0.4**2,
        mu=0.1,
        kappa=1.5,
        theta=0.4**2,
        sigma_v=0.6,
        rho=-0.5,
        lambda_jump=0.02,
        mu_jump=0.0,
        sigma_jump=0.05,
    )
    results = run_time_scale_sweep(
        model=model,
        days_list=[5, 10],
        n_paths=256,
        spot=30_000,
        target_price=31_000,
        base_seed=42,
        seed_offset=0,
    )
    assert [res.days for res in results] == [5, 10]
    assert results[0].delta_mean is None
    assert results[1].delta_mean is not None
    assert results[1].delta_up_probability is not None


if __name__ == "__main__":
    # Run the tests without requiring pytest
    test_parameter_estimation()
    test_model_simulation_shapes()
    test_statistics_output()
    test_time_scale_sweep()
    print("Sanity tests passed.")
