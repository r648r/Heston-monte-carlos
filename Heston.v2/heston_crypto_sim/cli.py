"""Command line interface orchestrating the full simulation workflow."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from .data_fetcher import BinanceDataFetcher
from .models.heston_jump import HestonJumpDiffusionModel
from .params_estimator import HestonParameterEstimator
from .polymarket_adapter import (
    compute_binary_market_probs,
    compute_bucket_market_probs,
)
from .regimes import detect_regime, estimate_jump_parameters
from .reporting.html_report import generate_report, save_report
from .reporting.plots import plot_distribution, plot_sample_paths
from .stats_tools import calculate_statistics, prob_updown
from .time_scale_analysis import (
    format_time_scale_table,
    run_time_scale_sweep,
    save_time_scale_results,
)

logger = logging.getLogger(__name__)


def _parse_bucket_ranges(bucket_args: Sequence[str] | None) -> list[tuple[float, float]]:
    if not bucket_args:
        return []
    buckets: list[tuple[float, float]] = []
    for item in bucket_args:
        try:
            low_str, high_str = item.split(":")
            low, high = float(low_str), float(high_str)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(f"Invalid bucket specification '{item}'. Expected low:high.") from exc
        buckets.append((low, high))
    return buckets


def _prepare_time_scale_days(days: Sequence[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for raw in days:
        day = int(raw)
        if day <= 0:
            raise ValueError("Time scale entries must be strictly positive integers.")
        if day not in seen:
            unique.append(day)
            seen.add(day)
    return unique


def _compute_forecast_days(days: int, target_date: str | None) -> tuple[int, datetime | None]:
    if target_date:
        try:
            target_dt = datetime.fromisoformat(target_date)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Target date must be ISO formatted, e.g. 2024-12-31.") from exc
        now = datetime.now(timezone.utc)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
        delta = (target_dt - now).days
        if delta <= 0:
            raise argparse.ArgumentTypeError("Target date must be in the future.")
        return delta, target_dt
    return days, None


def _check_extreme_paths(final_prices: np.ndarray, spot: float, forecast_days: int, threshold: float = 0.05) -> None:
    """Warn if many short-horizon paths explode or crash unreasonably."""
    if forecast_days > 10:
        return
    overdone = float(np.mean(final_prices > spot * 20))
    underdone = float(np.mean(final_prices < spot / 20))
    if overdone > threshold or underdone > threshold:
        logger.warning(
            "Detected heavy tails on a %s-day horizon (%.2f%% > 20×spot, %.2f%% < spot/20). Review parameters.",
            forecast_days,
            overdone * 100,
            underdone * 100,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heston Jump Diffusion simulator for crypto assets.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to simulate.")
    parser.add_argument("--paths", type=int, default=10_000, help="Number of Monte Carlo paths.")
    parser.add_argument("--days", type=int, default=30, help="Forecast horizon in calendar days.")
    parser.add_argument("--target-date", dest="target_date", help="ISO date overriding --days (e.g. 2024-12-31).")
    parser.add_argument("--target-price", type=float, help="Level used for P(S_T > target).")
    parser.add_argument("--vol-window", type=int, default=30, help="Window (days) for realized volatility.")
    parser.add_argument("--lookback-days", type=int, default=365, help="Amount of historical data to query.")
    parser.add_argument("--interval", default="1d", help="Binance kline interval (default: 1d).")
    parser.add_argument("--histogram-path", default="outputs/btc_distribution.png", help="Histogram output path.")
    parser.add_argument(
        "--paths-figure-path",
        default="outputs/mc_paths.png",
        help="Path to save the Monte Carlo sample paths figure.",
    )
    parser.add_argument(
        "--paths-figure-max",
        type=int,
        default=10_000,
        help="Maximum number of paths to plot (defaults to 10k).",
    )
    parser.add_argument("--output-html", default="outputs/bitcoin_heston_report.html", help="HTML report path.")
    parser.add_argument(
        "--thresholds-for-markets",
        nargs="*",
        type=float,
        default=None,
        help="List of strike levels for binary markets.",
    )
    parser.add_argument(
        "--bucket-ranges",
        nargs="*",
        help="Buckets in the form low:high; e.g. 30000:35000 35000:40000",
    )
    parser.add_argument(
        "--time-scale-sweep",
        nargs="*",
        type=int,
        help="Additional horizons (days) for comparative reliability analysis.",
    )
    parser.add_argument(
        "--time-scale-output",
        help="Optional JSON output path for --time-scale-sweep results.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        help="Seed for reproducible Monte Carlo draws.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        forecast_days, target_dt = _compute_forecast_days(args.days, args.target_date)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if forecast_days <= 0:
        parser.error("Forecast horizon must be strictly positive.")

    fetcher = BinanceDataFetcher()
    logger.info("Fetching current price for %s", args.symbol)
    current_price = fetcher.get_current_price(args.symbol)

    logger.info("Fetching %s days of historical data.", args.lookback_days)
    history = fetcher.get_historical_klines(
        symbol=args.symbol,
        interval=args.interval,
        lookback_days=args.lookback_days,
    )

    vol_annual, returns = fetcher.calculate_realized_volatility(history, window=args.vol_window)
    logger.info("Realized annualized volatility over window=%s: %.2f%%", args.vol_window, vol_annual * 100)

    regime_name, regime_factor = detect_regime(returns, window=args.vol_window)
    logger.info("Detected regime %s (factor %.2f).", regime_name, regime_factor)
    lambda_jump, mu_jump, sigma_jump = estimate_jump_parameters(returns)
    logger.info(
        "Jump parameters -> lambda=%.3f/yr, mu=%.4f, sigma=%.4f.",
        lambda_jump,
        mu_jump,
        sigma_jump,
    )

    estimator = HestonParameterEstimator()
    long_window = 180
    vol_of_vol_window = 30
    params = estimator.estimate_params(
        returns=returns,
        vol_annual=vol_annual,
        regime_name=regime_name,
        regime_factor=regime_factor,
        long_window=long_window,
        vol_window_for_volofvol=vol_of_vol_window,
    )

    S0 = current_price
    model = HestonJumpDiffusionModel(
        S0=S0,
        V0=params.V0,
        mu=params.mu,
        kappa=params.kappa,
        theta=params.theta,
        sigma_v=params.sigma_v,
        rho=params.rho,
        lambda_jump=lambda_jump,
        mu_jump=mu_jump,
        sigma_jump=sigma_jump,
    )

    logger.info("Running Monte Carlo with %s paths × %s steps.", args.paths, forecast_days)
    rng_main = np.random.default_rng(args.rng_seed) if args.rng_seed is not None else None
    sim_result = model.simulate_paths(
        T=forecast_days / 365,
        n_steps=max(forecast_days, 1),
        n_paths=args.paths,
        rng=rng_main,
    )
    final_prices = sim_result.prices[:, -1] if sim_result.prices.ndim == 2 else sim_result.prices
    if not np.all(np.isfinite(final_prices)):
        raise ValueError("Simulation output contains NaNs or infs; aborting.")
    if np.any(final_prices <= 0):
        raise ValueError("Simulation produced non-positive prices; aborting.")
    _check_extreme_paths(final_prices, current_price, forecast_days)

    observed_return_days = len(returns.dropna())
    theta_window = min(observed_return_days, long_window)
    model_parameters = [
        {
            "name": "S0",
            "value": f"${S0:,.2f}",
            "explanation": "Prix spot Binance utilisé comme point de départ des trajectoires.",
            "source": "dynamique",
        },
        {
            "name": "V0",
            "value": f"{params.V0:.6f}",
            "explanation": f"Variance initiale = vol annualisée ({vol_annual*100:.2f}%)² mesurée sur {args.vol_window} jours.",
            "source": "dynamique",
        },
        {
            "name": "mu",
            "value": f"{params.mu:.4f}",
            "explanation": f"Drift annuel = moyenne des rendements × 365 ajustée par le régime {regime_name.lower()} ({regime_factor:.2f}).",
            "source": "dynamique",
        },
        {
            "name": "kappa",
            "value": f"{params.kappa:.2f}",
            "explanation": "Vitesse de rappel définie heuristiquement selon le ratio vol actuelle / vol de long terme.",
            "source": "dynamique",
        },
        {
            "name": "theta",
            "value": f"{params.theta:.6f}",
            "explanation": f"Variance de long terme = moyenne des variances journalières sur {theta_window} jours × 365.",
            "source": "dynamique",
        },
        {
            "name": "sigma_v",
            "value": f"{params.sigma_v:.3f}",
            "explanation": f"Volatilité de la vol estimée via la dispersion des vols réalisés ({vol_of_vol_window} jours), bornée entre 0.3 et 1.5.",
            "source": "dynamique",
        },
        {
            "name": "rho",
            "value": f"{params.rho:.2f}",
            "explanation": "Corrélation returns/variations de volatilité (effet levier négatif imposé).",
            "source": "dynamique",
        },
        {
            "name": "lambda_jump",
            "value": f"{lambda_jump:.4f}",
            "explanation": "Intensité annualisée des mouvements extrêmes (|r| > 3σ) détectés.",
            "source": "dynamique",
        },
        {
            "name": "mu_jump",
            "value": f"{mu_jump:.4f}",
            "explanation": "Moyenne empirique des retours identifiés comme sauts.",
            "source": "dynamique",
        },
        {
            "name": "sigma_jump",
            "value": f"{sigma_jump:.4f}",
            "explanation": "Écart-type des retours classés comme sauts.",
            "source": "dynamique",
        },
        {
            "name": "regime_factor",
            "value": f"{regime_factor:.2f}",
            "explanation": f"Multiplicateur appliqué au drift pour refléter le régime {regime_name.lower()}.",
            "source": "dynamique",
        },
        {
            "name": "long_window",
            "value": f"{long_window}",
            "explanation": "Paramètre en dur : nombre de jours utilisés pour estimer la variance de long terme.",
            "source": "constant",
        },
        {
            "name": "vol_of_vol_window",
            "value": f"{vol_of_vol_window}",
            "explanation": "Paramètre en dur : fenêtre (jours) pour calculer la volatilité de la volatilité.",
            "source": "constant",
        },
    ]

    if target_dt is not None:
        end_dt = target_dt
        start_dt = target_dt - timedelta(days=forecast_days)
    else:
        start_dt = datetime.now(timezone.utc)
        end_dt = start_dt + timedelta(days=forecast_days)
    time_index = pd.date_range(start=start_dt, periods=forecast_days + 1, freq="D")

    stats = calculate_statistics(final_prices, target_price=args.target_price)
    up_probability = prob_updown(final_prices, current_price)

    thresholds = args.thresholds_for_markets or []
    binary_markets = (
        compute_binary_market_probs(final_prices, thresholds) if thresholds else {}
    )
    try:
        buckets = _parse_bucket_ranges(args.bucket_ranges)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    bucket_markets = compute_bucket_market_probs(final_prices, buckets) if buckets else {}

    if args.time_scale_sweep:
        sweep_candidates = [forecast_days] + list(args.time_scale_sweep)
        try:
            sweep_days = _prepare_time_scale_days(sweep_candidates)
        except ValueError as exc:
            parser.error(str(exc))
        sweep_results = run_time_scale_sweep(
            model=model,
            days_list=sweep_days,
            n_paths=args.paths,
            spot=current_price,
            target_price=args.target_price,
            base_seed=args.rng_seed,
            seed_offset=1,
        )
        summary_table = format_time_scale_table(sweep_results, include_target=args.target_price is not None)
        logger.info("Time-scale comparison:\n%s", summary_table)
        if args.time_scale_output:
            save_time_scale_results(sweep_results, args.time_scale_output)

    histogram_file = plot_distribution(final_prices, args.target_price, args.histogram_path)
    paths_figure = plot_sample_paths(
        sim_result.prices,
        args.paths_figure_path,
        timestamps=time_index,
        max_paths=args.paths_figure_max,
    )
    html_content = generate_report(
        stats,
        histogram_file,
        infos_contexte={
            "symbol": args.symbol,
            "current_price": current_price,
            "forecast_days": forecast_days,
            "n_paths": args.paths,
            "regime_name": regime_name,
            "regime_factor": regime_factor,
            "vol_annual": vol_annual,
            "binary_markets": binary_markets,
            "bucket_markets": bucket_markets,
            "up_probability": up_probability,
            "prob_above_ci": stats.get("prob_above_ci"),
            "model_parameters": model_parameters,
        },
        paths_image_path=paths_figure,
    )
    save_report(html_content, args.output_html)

    logger.info("Finished. Mean final price %.2f with 95%% CI [%s, %s].", stats["mean"], *stats["ci_95"])


if __name__ == "__main__":  # pragma: no cover
    main()
