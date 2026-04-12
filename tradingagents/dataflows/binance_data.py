"""Binance public market-data loader for crypto OHLCV.

Uses the Binance Vision public data API (https://data-api.binance.vision)
which is open, unauthenticated, and not geo-restricted. This module
replaces yfinance as the data source for crypto tickers, so that prices
match what users see on Binance rather than Yahoo's delayed aggregated
feed.

Symbol convention: this project's canonical crypto ticker is the pair
format ``BASE-USDT`` (e.g. ``BTC-USDT``, ``ETH-USDT``). Legacy
``BASE-USD`` inputs are also accepted and normalized to the equivalent
Binance symbol ``BASEUSDT``.
"""

import calendar
from datetime import datetime
from typing import Annotated, List, Optional

import pandas as pd
import requests

BINANCE_BASE = "https://data-api.binance.vision"
_KLINES_ENDPOINT = f"{BINANCE_BASE}/api/v3/klines"
_MAX_LIMIT = 1000  # Binance hard cap per klines request

# Map our interval → Binance interval (they largely overlap; keep explicit)
_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}


def normalize_binance_symbol(ticker: str) -> str:
    """Convert any crypto ticker variant to a Binance exchange symbol.

    Examples:
        BTC-USDT → BTCUSDT
        BTC-USD  → BTCUSDT  (legacy: USD is treated as USDT on Binance)
        BTC/USDT → BTCUSDT
        BTCUSDT  → BTCUSDT
    """
    t = ticker.upper().strip()
    t = t.replace("/", "-").replace(":", "-")
    if t.endswith("-USDT"):
        return t.replace("-USDT", "") + "USDT"
    if t.endswith("-USD"):
        return t.replace("-USD", "") + "USDT"
    # Already in Binance format (BTCUSDT) or unknown — pass through
    return t


def _to_binance_interval(interval: str) -> str:
    if interval not in _INTERVAL_MAP:
        raise ValueError(
            f"Unsupported interval '{interval}' for Binance. "
            f"Supported: {list(_INTERVAL_MAP.keys())}"
        )
    return _INTERVAL_MAP[interval]


def _fetch_klines(
    binance_symbol: str,
    binance_interval: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = _MAX_LIMIT,
) -> List[list]:
    """Call Binance klines API once. Returns raw kline rows."""
    params = {
        "symbol": binance_symbol,
        "interval": binance_interval,
        "limit": min(limit, _MAX_LIMIT),
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    resp = requests.get(_KLINES_ENDPOINT, params=params, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Binance klines request failed [{resp.status_code}]: {resp.text[:200]}"
        )
    return resp.json()


def _klines_to_dataframe(klines: List[list]) -> pd.DataFrame:
    """Convert raw Binance klines to a yfinance-compatible DataFrame."""
    if not klines:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def _fetch_recent_klines(
    binance_symbol: str, binance_interval: str, bars: int
) -> pd.DataFrame:
    """Fetch the most recent `bars` klines (paginated up to several pages)."""
    remaining = bars
    end_ms: Optional[int] = None
    collected: List[list] = []
    while remaining > 0:
        batch = _fetch_klines(
            binance_symbol,
            binance_interval,
            end_ms=end_ms,
            limit=min(remaining, _MAX_LIMIT),
        )
        if not batch:
            break
        collected = batch + collected
        remaining -= len(batch)
        # Step end_ms one millisecond before the earliest open_time to paginate back
        earliest_open = batch[0][0]
        end_ms = earliest_open - 1
        if len(batch) < _MAX_LIMIT:
            break
    return _klines_to_dataframe(collected)


def _fetch_range_klines(
    binance_symbol: str,
    binance_interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch all klines in [start_ms, end_ms], paginating forward."""
    collected: List[list] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = _fetch_klines(
            binance_symbol,
            binance_interval,
            start_ms=cursor,
            end_ms=end_ms,
            limit=_MAX_LIMIT,
        )
        if not batch:
            break
        collected.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < _MAX_LIMIT:
            break
        cursor = last_open + 1
    return _klines_to_dataframe(collected)


def load_binance_ohlcv(
    symbol: str, curr_date: str, interval: str = "1d"
) -> pd.DataFrame:
    """Load OHLCV from Binance with look-ahead-bias filtering.

    Mirrors the contract of ``stockstats_utils.load_ohlcv`` so it can be
    used as a drop-in vendor. Rows after ``curr_date`` are dropped.
    """
    from tradingagents.interval_utils import (
        get_default_lookback_bars,
        is_intraday,
    )

    bsym = normalize_binance_symbol(symbol)
    bint = _to_binance_interval(interval)
    curr_dt = pd.to_datetime(curr_date)

    # Intraday safety: if the caller passed a bare date (midnight UTC) AND
    # that date is today or in the future, promote the cutoff to "now".
    # Otherwise indicators would be truncated at 00:00 UTC and the LLM would
    # report hours-stale prices as "current".
    if is_intraday(interval) and curr_dt.hour == 0 and curr_dt.minute == 0:
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        if curr_dt.date() >= now_utc.date():
            curr_dt = now_utc

    # How many bars to fetch:
    #  - Daily: ~5 years ≈ 1825 bars → multiple pages
    #  - Intraday: ~3x the default lookback bars so indicators with
    #    longer warmup (50 SMA, VWMA etc.) have enough history
    if is_intraday(interval):
        bars_needed = max(get_default_lookback_bars(interval) * 4, 300)
    else:
        bars_needed = 1825

    df = _fetch_recent_klines(bsym, bint, bars_needed)

    if df.empty:
        return df

    # Drop bars strictly after curr_date (no look-ahead bias)
    df = df[df["Date"] <= curr_dt].reset_index(drop=True)
    return df


def get_binance_stock_data(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USDT"],
    start_date: Annotated[str, "Start date YYYY-MM-DD or YYYY-MM-DD HH:MM"],
    end_date: Annotated[str, "End date YYYY-MM-DD or YYYY-MM-DD HH:MM"],
    interval: str = "1d",
) -> str:
    """Binance implementation of get_stock_data. Returns a CSV string
    matching yfinance's output format so downstream prompts are unchanged."""
    from tradingagents.interval_utils import parse_trade_datetime

    bsym = normalize_binance_symbol(symbol)
    bint = _to_binance_interval(interval)

    # Naive datetimes are interpreted as UTC to match the convention used
    # elsewhere in the project (yfinance returns tz-stripped UTC for crypto,
    # and the trading_graph passes user-typed datetimes through unchanged).
    # Using calendar.timegm avoids the local-time assumption of .timestamp().
    start_dt = parse_trade_datetime(start_date)
    end_dt = parse_trade_datetime(end_date)

    # Intraday safety: if end_date is a bare date (midnight UTC) on today or
    # a future day, extend to "now" so the caller sees the most recent bars.
    # Same for same-day start==end queries where start is also midnight.
    from datetime import datetime as _dt, timezone as _tz
    if interval != "1d":
        now_utc = _dt.now(_tz.utc).replace(tzinfo=None)
        if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.date() >= now_utc.date():
            end_dt = now_utc
        if (
            start_dt.date() == end_dt.date()
            and start_dt.hour == 0
            and start_dt.minute == 0
        ):
            # Caller wanted "today's intraday bars" — start from beginning of day
            pass  # keep start_dt at midnight

    start_ms = calendar.timegm(start_dt.timetuple()) * 1000
    end_ms = calendar.timegm(end_dt.timetuple()) * 1000

    df = _fetch_range_klines(bsym, bint, start_ms, end_ms)

    if df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    df = df.set_index("Date")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)

    csv_string = df.to_csv()
    from datetime import timezone
    header = (
        f"# Stock data for {symbol.upper()} (Binance {bsym}, interval={interval}) "
        f"from {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Timestamps are UTC; Date column is bar open_time\n"
        f"# Data retrieved on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
    )
    return header + csv_string


def get_binance_indicators_window(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "1d",
) -> str:
    """Binance implementation of get_stock_stats_indicators_window.

    Delegates to the shared stockstats-based calculator but forces the
    data vendor to Binance so indicators are computed from Binance OHLCV.
    """
    from .y_finance import get_stock_stats_indicators_window

    return get_stock_stats_indicators_window(
        symbol=symbol,
        indicator=indicator,
        curr_date=curr_date,
        look_back_days=look_back_days,
        interval=interval,
        vendor="binance",
    )
