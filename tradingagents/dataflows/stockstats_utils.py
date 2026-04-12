import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(
    symbol: str,
    curr_date: str,
    interval: str = "1d",
    vendor: str = "yfinance",
) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads historical data and caches per symbol/interval. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.

    For daily data: downloads 5 years of history.
    For intraday data: downloads max allowed period per yfinance limits.

    When ``vendor="binance"`` the call is delegated to
    ``binance_data.load_binance_ohlcv`` instead of yfinance. Binance data
    is fetched fresh per call (no disk cache) because Binance public
    endpoints are fast and rate-limit friendly.
    """
    from tradingagents.interval_utils import is_intraday, get_yf_max_period

    if vendor == "binance":
        from .binance_data import load_binance_ohlcv
        return load_binance_ohlcv(symbol, curr_date, interval=interval)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    os.makedirs(config["data_cache_dir"], exist_ok=True)

    if is_intraday(interval):
        # Intraday: cache per symbol + interval + today's date
        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-{interval}-{today_str}.csv",
        )

        if os.path.exists(data_file):
            data = pd.read_csv(data_file, on_bad_lines="skip")
        else:
            period = get_yf_max_period(interval)
            data = yf_retry(lambda: yf.download(
                symbol,
                period=period,
                interval=interval,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            ))
            data = data.reset_index()
            # Rename Datetime column (used by intraday) to Date for consistency
            if "Datetime" in data.columns:
                data = data.rename(columns={"Datetime": "Date"})
            data.to_csv(data_file, index=False)
    else:
        # Daily: cache uses a fixed window (5y to today) so one file per symbol
        today_date = pd.Timestamp.today()
        start_date = today_date - pd.DateOffset(years=5)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = today_date.strftime("%Y-%m-%d")

        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
        )

        if os.path.exists(data_file):
            data = pd.read_csv(data_file, on_bad_lines="skip")
        else:
            data = yf_retry(lambda: yf.download(
                symbol,
                start=start_str,
                end=end_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            ))
            data = data.reset_index()
            data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)

    # Strip timezone info for consistent comparison
    if data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_localize(None)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        interval: str = "1d",
    ):
        from tradingagents.interval_utils import is_intraday, datetime_format

        data = load_ohlcv(symbol, curr_date, interval=interval)
        df = wrap(data)

        dt_fmt = datetime_format(interval)
        df["Date"] = df["Date"].dt.strftime(dt_fmt)

        curr_date_str = pd.to_datetime(curr_date).strftime(dt_fmt)

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
