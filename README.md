# Heston Monte-Carlo — Crypto Stochastic-Volatility Simulator

**English** · [Français](README.fr.md)

Quantitative-finance project: a **Heston stochastic-volatility model with jumps**
applied to crypto markets, plus a learning lab that builds the theory from scratch.

## `Heston.v2/` — Bitcoin Heston simulator (flagship)

A clean, modular, tested Python package (`heston_crypto_sim`) that:

- pulls market data from the **Binance** public API (`data_fetcher`)
- detects the current **market regime** (`regimes`)
- estimates **Heston + jump** parameters heuristically (`params_estimator`)
- runs a **Monte-Carlo** simulation of price paths (`models/heston_jump`)
- computes probabilities for **Polymarket**-style bets — up/down,
  "above K", "inside [K1, K2]" (`stats_tools`, `polymarket_adapter`)
- generates plots and a standalone **HTML report** (`reporting/`)

```bash
cd Heston.v2
python -m heston_crypto_sim.cli      # run a simulation + HTML report
pytest tests_sanity.py               # sanity tests
```

### Sample output

Simulated BTC price paths and the resulting terminal-price distribution:

![Monte-Carlo simulated price paths](Heston.v2/outputs/nov_paths.png)

![Terminal price distribution](Heston.v2/outputs/nov_hist.png)

## `heston-learning-lab/` — step-by-step theory (FR notebooks)

Dockerised Jupyter lab (5 notebooks, in French) building the model from the
ground up: Brownian motion -> full Heston model -> Monte-Carlo -> HTML report
generation.

```bash
cd heston-learning-lab && ./start_lab.sh   # Jupyter via Docker
```

## `kyle/` — GBM Monte-Carlo
Geometric-Brownian-Motion portfolio simulations (warm-up scripts).

---
*Educational / research project — not financial advice.*
