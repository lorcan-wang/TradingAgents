"""Utility functions for translating trading_interval config into
parameters needed by data providers, caching, prompts, etc."""

from datetime import datetime

VALID_INTERVALS = ("1m", "5m", "15m", "30m", "1h", "1d")


def is_intraday(interval: str) -> bool:
    return interval != "1d"


def get_yf_interval(interval: str) -> str:
    """Return yfinance-compatible interval string (identity mapping)."""
    return interval


def get_yf_max_period(interval: str) -> str:
    """Max lookback period allowed by yfinance for each interval."""
    return {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "1d": "5y",
    }.get(interval, "5y")


def get_av_interval(interval: str) -> str:
    """Map config interval to Alpha Vantage interval string."""
    return {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
        "1d": "daily",
    }.get(interval, "daily")


def get_av_function(interval: str) -> str:
    """Return the Alpha Vantage time-series function name."""
    if is_intraday(interval):
        return "TIME_SERIES_INTRADAY"
    return "TIME_SERIES_DAILY_ADJUSTED"


def get_default_lookback_bars(interval: str) -> int:
    """Sensible default number of bars to look back for indicators."""
    return {
        "1m": 60,
        "5m": 120,
        "15m": 80,
        "30m": 60,
        "1h": 50,
        "1d": 30,
    }.get(interval, 30)


def get_news_lookback_days(interval: str) -> int:
    """How many days of news to fetch."""
    return 1 if is_intraday(interval) else 7


def get_cache_max_days(interval: str) -> int:
    """Maximum days of data to cache per interval."""
    return {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "1d": 1825,
    }.get(interval, 1825)


def get_prompt_timeframe_context(interval: str) -> str:
    """Return a context string appended to analyst prompts for intraday mode.

    Returns empty string for daily mode so existing prompts are unchanged.
    """
    if not is_intraday(interval):
        return ""

    labels = {
        "1m": "1-minute",
        "5m": "5-minute",
        "15m": "15-minute",
        "30m": "30-minute",
        "1h": "1-hour",
    }
    label = labels.get(interval, interval)

    return (
        f"\n\n[INTRADAY MODE] You are analyzing {label} intraday price data. "
        "Focus on short-term momentum, volume patterns, and intraday price action. "
        "Prefer short-period indicators (EMA, RSI, MACD, Bollinger Bands) over "
        "long-term indicators (200 SMA). The trading horizon is within a single "
        "trading session."
    )


def parse_trade_datetime(trade_date: str) -> datetime:
    """Parse trade_date supporting both daily and intraday formats."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(trade_date, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid trade_date format: '{trade_date}'. "
        "Expected YYYY-MM-DD or YYYY-MM-DD HH:MM"
    )


def datetime_format(interval: str) -> str:
    """Return the datetime format string appropriate for the interval."""
    if is_intraday(interval):
        return "%Y-%m-%d %H:%M"
    return "%Y-%m-%d"
