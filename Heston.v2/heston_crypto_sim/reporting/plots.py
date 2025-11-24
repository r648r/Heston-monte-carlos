"""Plotting utilities for Monte Carlo distributions."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats_tools import ensure_numpy_array

logger = logging.getLogger(__name__)


def plot_distribution(
    final_prices: np.ndarray,
    target_price: float | None,
    output_path: str | Path,
    bins: int = 60,
) -> Path:
    """Create and save a histogram highlighting key thresholds."""
    prices = ensure_numpy_array(final_prices)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(prices, bins=bins, color="#5B8BF7", alpha=0.75)
    ax.set_xlabel("Simulated final price")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of simulated final prices")

    mean_val = float(np.mean(prices))
    ax.axvline(
        mean_val,
        color="#111827",
        linestyle="--",
        linewidth=2,
        label=f"Moyenne attendue = {mean_val:,.2f}",
    )

    if target_price is not None:
        ax.axvline(
            target_price,
            color="#ef4444",
            linestyle=":",
            linewidth=2,
            label=f"Seuil ciblé = {target_price:,.2f}",
        )

    percentile_levels = (5, 25, 50, 75, 95)
    percentile_colors = {
        5: "#ea580c",
        25: "#fbbf24",
        50: "#3b82f6",
        75: "#22c55e",
        95: "#a855f7",
    }
    percentile_values = {p: float(np.percentile(prices, p)) for p in percentile_levels}
    for p, value in percentile_values.items():
        color = percentile_colors.get(p, "#94a3b8")
        ax.axvline(value, color=color, linestyle="-.", linewidth=1.4, alpha=0.9)
        ax.text(
            value,
            ax.get_ylim()[1] * 0.92,
            f"P{p}",
            rotation=90,
            color=color,
            fontsize=9,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved distribution plot to %s", output_path)
    return output_path


def plot_sample_paths(
    price_paths: np.ndarray,
    output_path: str | Path,
    timestamps: Sequence[datetime] | pd.DatetimeIndex | None = None,
    max_paths: int | None = None,
    hist_bins: int = 25,
) -> Path:
    """Plot Monte Carlo trajectories with a commented histogram of final prices."""
    prices = np.asarray(price_paths, dtype=float)
    if prices.ndim != 2:
        raise ValueError("price_paths must be a 2D array of shape (n_paths, n_steps+1).")
    if prices.size == 0:
        raise ValueError("price_paths cannot be empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_paths = prices.shape[0] if max_paths is None else min(max_paths, prices.shape[0])

    if timestamps is not None:
        time_index = pd.to_datetime(list(timestamps))
        if len(time_index) != prices.shape[1]:
            raise ValueError("Length of timestamps must match number of time steps.")
    else:
        time_index = np.arange(prices.shape[1])

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(1, 16, wspace=0.05)
    ax_paths = fig.add_subplot(grid[0, :12])
    ax_hist = fig.add_subplot(grid[0, 12:], sharey=ax_paths)
    if n_paths < prices.shape[0]:
        idx_options = np.linspace(0, prices.shape[0] - 1, n_paths, dtype=int)
        idx_to_plot = np.unique(idx_options)
    else:
        idx_to_plot = np.arange(prices.shape[0])
    colors = plt.cm.plasma(np.linspace(0, 1, len(idx_to_plot)))

    for idx, color in zip(idx_to_plot, colors):
        ax_paths.plot(
            time_index,
            prices[idx],
            color=color,
            linewidth=0.7,
            alpha=0.75,
        )

    final_percentiles = {
        10: float(np.percentile(prices[:, -1], 10)),
        25: float(np.percentile(prices[:, -1], 25)),
        50: float(np.percentile(prices[:, -1], 50)),
        75: float(np.percentile(prices[:, -1], 75)),
        90: float(np.percentile(prices[:, -1], 90)),
    }
    percentile_colors = {
        10: "#ea580c",
        25: "#f97316",
        50: "#3b82f6",
        75: "#22c55e",
        90: "#a855f7",
    }

    for percentile, value in final_percentiles.items():
        color = percentile_colors.get(percentile, "#94a3b8")
        ax_paths.axhline(value, color=color, linestyle="--", linewidth=1.2)
        ax_paths.text(
            time_index[0],
            value,
            f"P{percentile}: {value:,.0f}",
            color=color,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

    n_plotted = len(idx_to_plot)
    ax_paths.set_title(f"Monte Carlo Paths ({n_plotted}/{prices.shape[0]} plotted)")
    ax_paths.set_xlabel("Date" if timestamps is not None else "Time step index")
    ax_paths.set_ylabel("Price")

    final_prices = prices[:, -1]
    hist_color = "#1d4ed8"
    counts, bins_edges, patches = ax_hist.hist(
        final_prices,
        bins=hist_bins,
        orientation="horizontal",
        color=hist_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    mean_final = float(np.mean(final_prices))
    mean_line = ax_hist.axhline(
        mean_final,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {mean_final:,.0f}",
    )
    median_final = float(np.median(final_prices))
    median_line = ax_hist.axhline(
        median_final,
        color="#0f766e",
        linestyle=":",
        linewidth=1.1,
        label=f"Median: {median_final:,.0f}",
    )

    total_paths = float(final_prices.size)
    max_count = counts.max() if counts.size else 0.0
    label_offset = max_count * 0.02
    for count, patch in zip(counts, patches):
        if count <= 0:
            continue
        percent = (count / total_paths) * 100 if total_paths > 0 else 0.0
        y_center = patch.get_y() + patch.get_height() / 2
        ax_hist.text(
            patch.get_width() + label_offset,
            y_center,
            f"{percent:.1f}%",
            color="#1e293b",
            fontsize=8,
            va="center",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax_hist.set_xlabel("Frequency")
    ax_hist.set_title("Final price histogram", fontsize=11)
    ax_hist.grid(False)
    ax_hist.yaxis.tick_right()
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    if counts.size:
        ax_hist.legend(handles=[mean_line, median_line], loc="lower right", frameon=False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved Monte Carlo paths plot to %s", output_path)
    return output_path
