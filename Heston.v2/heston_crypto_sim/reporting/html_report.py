"""HTML reporting utilities for the simulation outputs."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..polymarket_adapter import ProbabilityEstimate

logger = logging.getLogger(__name__)


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def generate_report(
    stats: dict,
    histogram_path: str | Path,
    infos_contexte: dict[str, Any],
    paths_image_path: str | Path | None = None,
) -> str:
    """Return a styled HTML report summarizing the simulation."""
    histogram_path = Path(histogram_path)
    histogram_b64 = _encode_image(histogram_path)
    paths_b64 = _encode_image(Path(paths_image_path)) if paths_image_path else None
    ci_lower, ci_upper = stats["ci_95"]
    prob_above = stats.get("prob_above_target")
    prob_above_ci = stats.get("prob_above_ci")
    target_price = stats.get("target_price")

    context_defaults = {
        "symbol": "BTCUSDT",
        "current_price": float("nan"),
        "forecast_days": 0,
        "n_paths": 0,
        "regime_name": "UNKNOWN",
        "vol_annual": float("nan"),
        "regime_factor": 1.0,
    }
    ctx = {**context_defaults, **infos_contexte}

    binary_markets = ctx.get("binary_markets", {})
    bucket_markets = ctx.get("bucket_markets", {})

    percentiles_rows = "\n".join(
        f"<tr><td>{p}th</td><td>{value:,.2f}</td></tr>"
        for p, value in stats["percentiles"].items()
    )
    def _format_probability_cell(prob_entry: ProbabilityEstimate | float) -> tuple[float, tuple[float, float]]:
        if isinstance(prob_entry, ProbabilityEstimate):
            return prob_entry.probability, prob_entry.ci
        return float(prob_entry), (float(prob_entry), float(prob_entry))

    binary_rows_list: list[str] = []
    for k, estimate in binary_markets.items():
        prob_value, (ci_low, ci_high) = _format_probability_cell(estimate)
        phrase = (
            f"{prob_value*100:.2f}% ({ci_low*100:.2f}%-{ci_high*100:.2f}%) des chemins dépassent {k:,.0f} USD."
        )
        binary_rows_list.append(
            f"<tr title=\"{phrase}\"><td title=\"{phrase}\">{k:,.0f}</td>"
            f"<td title=\"{phrase}\">{prob_value*100:.2f}%<br/><small>[{ci_low*100:.2f}%, {ci_high*100:.2f}%]</small></td></tr>"
        )
    binary_rows = "\n".join(binary_rows_list)

    bucket_rows_list: list[str] = []
    for (low, high), estimate in bucket_markets.items():
        prob_value, (ci_low, ci_high) = _format_probability_cell(estimate)
        phrase = (
            f"{prob_value*100:.2f}% ({ci_low*100:.2f}%-{ci_high*100:.2f}%) des chemins terminent entre {low:,.0f} et {high:,.0f} USD."
        )
        bucket_rows_list.append(
            f"<tr title=\"{phrase}\"><td title=\"{phrase}\">[{low:,.0f}, {high:,.0f})</td>"
            f"<td title=\"{phrase}\">{prob_value*100:.2f}%<br/><small>[{ci_low*100:.2f}%, {ci_high*100:.2f}%]</small></td></tr>"
        )
    bucket_rows = "\n".join(bucket_rows_list)
    params_table = ctx.get("model_parameters", [])
    params_rows = "\n".join(
        f"<tr><td>{item['name']}</td><td>{item['value']}</td><td>{item.get('source', 'n/a')}</td><td><em>{item['explanation']}</em></td></tr>"
        for item in params_table
    )

    target_section = ""
    if prob_above is not None and target_price is not None:
        ci_display = ""
        if prob_above_ci is not None:
            ci_low, ci_high = prob_above_ci
            ci_display = f"<div class=\"stat-sub\">IC 95%: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]</div>"
        target_section = f"""
            <div class="stat-card" title="Probabilité modelisée de dépasser le seuil cible à l'échéance.">
                <div class="stat-label">P(S_T &gt; {target_price:,.0f})</div>
                <div class="stat-value highlight">{prob_above*100:.3f}%</div>
                {ci_display}
                <p class="card-note">*Lecture directe pour un pari “Above {target_price:,.0f}”*</p>
            </div>
        """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Heston Crypto Forecast - {datetime.utcnow():%Y-%m-%d %H:%M UTC}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 40px; }}
.container {{ max-width: 1100px; margin: auto; background: #1e293b; border-radius: 18px; overflow: hidden; box-shadow: 0 15px 40px rgba(0,0,0,0.4); }}
.header {{ background: linear-gradient(135deg, #2563eb, #7c3aed); padding: 30px; }}
.header h1 {{ margin: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; padding: 30px; }}
.stat-card {{ background: #0f172a; padding: 20px; border-radius: 12px; border: 1px solid rgba(148,163,184,0.2); position: relative; }}
.stat-label {{ font-size: 0.9em; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; }}
.stat-value {{ font-size: 1.8em; font-weight: 600; }}
.stat-value.highlight {{ color: #4ade80; }}
.card-note {{ font-style: italic; color: #94a3b8; font-size: 0.8em; margin-top: 8px; }}
.section {{ padding: 30px; }}
.section h2 {{ margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; border-radius: 10px; overflow: hidden; }}
th {{ background: rgba(59,130,246,0.15); text-transform: uppercase; letter-spacing: 0.05em; }}
th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid rgba(148,163,184,0.1); }}
tbody tr:nth-child(odd) {{ background: rgba(15,23,42,0.35); }}
tbody tr:hover {{ background: rgba(96,165,250,0.15); }}
.image {{ text-align: center; }}
.image img {{ width: 100%; border-radius: 12px; }}
.badge {{ display: inline-block; padding: 6px 14px; border-radius: 999px; background: #475569; text-transform: uppercase; font-size: 0.8em; }}
.section-note {{ font-style: italic; color: #94a3b8; margin-top: 12px; }}
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Heston Jump Diffusion Forecast</h1>
            <p>Symbol {ctx['symbol']} · Regime <span class="badge">{ctx['regime_name']}</span></p>
        </div>
        <div class="grid">
            <div class="stat-card" title="Prix spot obtenu via Binance, point de départ de la trajectoire simulée.">
                <div class="stat-label">Current Price</div>
                <div class="stat-value">${ctx['current_price']:,.2f}</div>
                <p class="card-note">*Base de comparaison pour toutes les projections*</p>
            </div>
            <div class="stat-card" title="Nombre de jours entre aujourd'hui et la date cible du marché.">
                <div class="stat-label">Forecast Horizon</div>
                <div class="stat-value">{ctx['forecast_days']} days</div>
                <p class="card-note">*Durée pendant laquelle la dynamique Heston est intégrée*</p>
            </div>
            <div class="stat-card" title="Nombre de trajectoires Monte Carlo générées pour estimer la distribution.">
                <div class="stat-label">Simulations</div>
                <div class="stat-value">{ctx['n_paths']:,}</div>
                <p class="card-note">*Plus il y en a, plus la distribution est stable*</p>
            </div>
            <div class="stat-card" title="Volatilité annualisée (réalisée) utilisée comme V0 et référence pour les paramètres.">
                <div class="stat-label">Annualized Vol</div>
                <div class="stat-value">{ctx['vol_annual']*100:.2f}%</div>
                <p class="card-note">*Mesure instantanée du stress de marché*</p>
            </div>
            {target_section}
        </div>
        <div class="section image">
            <h2>Distribution of Final Prices</h2>
            <img src="data:image/png;base64,{histogram_b64}" alt="Distribution Histogram" />
            <p>95% interval: [{ci_lower:,.2f}, {ci_upper:,.2f}] · Mean price: {stats["mean"]:,.2f}</p>
            <p class="section-note"><em>La ligne noire pointillée indique la moyenne attendue, les lignes colorées signalent les percentiles clés, et la ligne rouge rappelle le seuil de pari.</em></p>
        </div>
        {f'''
        <div class="section image">
            <h2>Sample Monte Carlo Paths</h2>
            <img src="data:image/png;base64,{paths_b64}" alt="Monte Carlo sample paths" />
            <p>The chart displays up to {infos_contexte.get("n_paths", 0)} simulated paths (capped for legibility).</p>
            <p class="section-note"><em>Chaque courbe part du spot Binance; par exemple «P90: 94 781» signifie que 90% des trajectoires terminent sous 94 781 USD (et 10% au-dessus), ce qui permet de visualiser directement l'étendue finale.</em></p>
        </div>''' if paths_b64 else ''}
        <div class="section">
            <h2>Paramètres du modèle Heston</h2>
            <p class="section-note"><em>Chaque paramètre est calibré de façon heuristique à partir des données récentes : spot, volatilités, régime détecté et statistique des sauts.</em></p>
            <table class="data-table params-table">
                <thead><tr><th>Paramètre</th><th>Valeur</th><th>Origine</th><th>Explication</th></tr></thead>
                <tbody>{params_rows or '<tr><td colspan="4">Paramètres non disponibles</td></tr>'}</tbody>
            </table>
        </div>
        <div class="section">
            <h2>Percentiles</h2>
            <table class="data-table percentile-table">
                <thead><tr><th>Percentile</th><th>Price</th></tr></thead>
                <tbody>
                    {percentiles_rows}
                </tbody>
            </table>
            <p class="section-note"><em>Les quantiles bas (P5, P25) décrivent les scénarios baissiers, la médiane (P50) le cas central, et les quantiles élevés (P75, P95) illustrent la queue droite des hausses possibles.</em></p>
        </div>
        <div class="section">
            <h2>Binary Markets P(S_T &gt; K)</h2>
            <p class="section-note"><em>Comparer ces probabilités internes aux cotes Polymarket pour estimer un edge potentiel sur chaque strike.</em></p>
            <table class="data-table table-binary">
                <thead><tr><th>K</th><th>Probability</th></tr></thead>
                <tbody>{binary_rows or '<tr><td colspan="2">No thresholds supplied</td></tr>'}</tbody>
            </table>
        </div>
        <div class="section">
            <h2>Bucket Markets P(K1 ≤ S_T &lt; K2)</h2>
            <p class="section-note"><em>Les buckets illustrent la masse de distribution par intervalles : utile pour les marchés range ou “inside/outside”.</em></p>
            <table class="data-table table-bucket">
                <thead><tr><th>Bucket</th><th>Probability</th></tr></thead>
                <tbody>{bucket_rows or '<tr><td colspan="2">No buckets supplied</td></tr>'}</tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """
    return html_content


def save_report(html_content: str, output_path: str | Path) -> Path:
    """Persist the generated HTML content to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info("Saved HTML report to %s", output_path)
    return output_path
