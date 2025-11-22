"""Data acquisition helpers for Binance crypto data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class BinanceAPIError(RuntimeError):
    """Raised when the Binance API returns an error payload."""


class BinanceDataFetcher:
    """Utility class that wraps Binance REST endpoints with error handling."""

    BASE_URL = "https://api.binance.com/api/v3"
    MAX_LIMIT = 1000

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, endpoint: str, params: dict[str, Any]) -> Any:
        """Call Binance REST API with retries and basic validation."""
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and "code" in payload and payload["code"] != 200:
                    raise BinanceAPIError(payload.get("msg", "Unknown Binance API error"))
                return payload
            except requests.RequestException as exc:  # pragma: no cover - network errors
                logger.warning(
                    "Binance API request failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise BinanceAPIError(f"Failed to call Binance API endpoint {endpoint}") from exc
        raise BinanceAPIError(f"Failed to call Binance API endpoint {endpoint}")

    def get_current_price(self, symbol: str) -> float:
        """Return the latest price for a given Binance symbol."""
        payload = self._request("ticker/price", {"symbol": symbol})
        try:
            return float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise BinanceAPIError(f"Unexpected payload when fetching ticker for {symbol}") from exc

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Retrieve historical klines for the requested symbol."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        all_rows: list[list[Any]] = []
        next_start = start_ms

        while next_start < end_ms and len(all_rows) < lookback_days + 5:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": next_start,
                "limit": self.MAX_LIMIT,
            }
            batch = self._request("klines", params)
            if not batch:
                break
            all_rows.extend(batch)
            last_close = int(batch[-1][6])
            if last_close >= end_ms or len(batch) < self.MAX_LIMIT:
                break
            next_start = last_close + 1

        if not all_rows:
            raise BinanceAPIError(f"No kline data returned for {symbol}")

        df = pd.DataFrame(
            all_rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        df = df.sort_values("open_time").reset_index(drop=True)
        if len(df) > lookback_days:
            df = df.tail(lookback_days).reset_index(drop=True)

        return df

    def calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        window: int = 30,
    ) -> tuple[float, pd.Series]:
        """Compute log returns and the latest annualized realized volatility."""
        if "close" not in df or df["close"].isna().all():
            raise ValueError("Price DataFrame must contain a non-empty 'close' column.")

        close_series = df["close"].astype(float)
        returns = np.log(close_series / close_series.shift(1)).dropna()
        rolling_std = returns.rolling(window=window).std()
        if rolling_std.dropna().empty:
            logger.warning("Not enough data for realized volatility window=%s; using simple std.", window)
            vol_annual = float(returns.std() * np.sqrt(365))
        else:
            vol_annual = float(rolling_std.iloc[-1] * np.sqrt(365))

        if not np.isfinite(vol_annual):
            raise ValueError("Computed realized volatility is not finite.")

        return vol_annual, returns
