"""Utilities to run the Heston simulator across multiple time scales."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from .models.heston_jump import HestonJumpDiffusionModel
from .stats_tools import calculate_statistics, prob_updown

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimeScaleResult:
    """Summary statistics for a single forecast horizon."""

    days: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float
    se_mean: float
    up_probability: float
    prob_above_target: float | None
    prob_above_ci: tuple[float, float] | None
    reliability_flags: list[str]
    delta_mean: float | None = None
    delta_up_probability: float | None = None
    delta_target_probability: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _reliability_flags(
    stats: dict,
    n_paths: int,
    spot: float,
    horizon_days: int,
) -> list[str]:
    flags: list[str] = []
    if n_paths < 2_000:
        flags.append("paths<2k")

    se_ratio = stats["se_mean"] / max(abs(stats["mean"]), 1e-12)
    if se_ratio > 0.2:
        flags.append(f"SE/mean>{se_ratio:.0%}")

    ci_range = stats["ci_95"][1] - stats["ci_95"][0]
    if spot > 0:
        ci_ratio = ci_range / spot
        if ci_ratio > 0.8:
            flags.append(f"CI width>{ci_ratio:.0%} spot")

    std_ratio = stats["std"] / max(spot, 1e-12)
    if std_ratio > 1.0 and horizon_days <= 14:
        flags.append(f"std>{std_ratio:.0%} spot on short horizon")

    prob_ci = stats.get("prob_above_ci")
    if prob_ci:
        width = prob_ci[1] - prob_ci[0]
        if width > 0.3:
            flags.append(f"prob CI width {width:.0%}")

    return flags


def _prepare_days(days_list: Sequence[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for raw in days_list:
        day = int(raw)
        if day <= 0:
            raise ValueError("Forecast horizons must be strictly positive integers.")
        if day not in seen:
            unique.append(day)
            seen.add(day)
    return unique


def run_time_scale_sweep(
    model: HestonJumpDiffusionModel,
    days_list: Sequence[int],
    n_paths: int,
    spot: float,
    target_price: float | None = None,
    base_seed: int | None = None,
    seed_offset: int = 0,
) -> list[TimeScaleResult]:
    """Simulate several horizons and capture reliability diagnostics."""
    horizons = _prepare_days(days_list)
    results: list[TimeScaleResult] = []

    for idx, days in enumerate(horizons):
        rng_seed = None if base_seed is None else base_seed + seed_offset + idx
        rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        sim = model.simulate_paths(
            T=days / 365,
            n_steps=max(days, 1),
            n_paths=n_paths,
            rng=rng,
        )
        final_prices = sim.prices[:, -1] if sim.prices.ndim == 2 else sim.prices
        stats = calculate_statistics(final_prices, target_price=target_price)
        flags = _reliability_flags(stats, n_paths, spot, days)
        target_prob = stats.get("prob_above_target")
        result = TimeScaleResult(
            days=days,
            mean=float(stats["mean"]),
            median=float(stats["percentiles"]["50"]),
            std=float(stats["std"]),
            ci_low=float(stats["ci_95"][0]),
            ci_high=float(stats["ci_95"][1]),
            se_mean=float(stats["se_mean"]),
            up_probability=float(prob_updown(final_prices, spot)),
            prob_above_target=float(target_prob) if target_prob is not None else None,
            prob_above_ci=stats.get("prob_above_ci"),
            reliability_flags=flags,
        )
        results.append(result)

    _attach_deltas(results)
    return results


def _attach_deltas(results: list[TimeScaleResult]) -> None:
    if not results:
        return
    base = results[0]
    for res in results[1:]:
        res.delta_mean = res.mean - base.mean
        res.delta_up_probability = res.up_probability - base.up_probability
        if res.prob_above_target is not None and base.prob_above_target is not None:
            res.delta_target_probability = res.prob_above_target - base.prob_above_target


def format_time_scale_table(
    results: Sequence[TimeScaleResult],
    include_target: bool,
) -> str:
    """Return a formatted string describing the sweep for logs/CLI."""
    if not results:
        return "No time-scale results."

    header = ["Days", "Mean", "Std", "UpProb", "ΔUp", "SE/Mean", "CI95"]
    if include_target:
        header.extend(["P>target", "ΔTarget"])
    header.append("Flags")
    lines = [" | ".join(header)]
    lines.append("-" * len(lines[0]))
    for res in results:
        se_ratio = res.se_mean / max(abs(res.mean), 1e-12)
        row = [
            f"{res.days:4d}",
            f"{res.mean:,.2f}",
            f"{res.std:,.2f}",
            f"{res.up_probability:.3f}",
            f"{res.delta_up_probability:+.3f}" if res.delta_up_probability is not None else "base",
            f"{se_ratio:.1%}",
            f"[{res.ci_low:,.0f}, {res.ci_high:,.0f}]",
        ]
        if include_target:
            if res.prob_above_target is None:
                row.extend(["-", "-"])
            else:
                row.append(f"{res.prob_above_target:.3f}")
                row.append(
                    f"{res.delta_target_probability:+.3f}"
                    if res.delta_target_probability is not None
                    else "base"
                )
        row.append("; ".join(res.reliability_flags) if res.reliability_flags else "OK")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def save_time_scale_results(results: Iterable[TimeScaleResult], output_path: str) -> str:
    """Persist the sweep metrics as JSON for later analysis."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    serialisable = [res.to_dict() for res in results]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Saved time-scale sweep results to %s", output_path)
    return output_path
