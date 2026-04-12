from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "Current trading time. Daily: 'YYYY-MM-DD'. Intraday: 'YYYY-MM-DD HH:MM' (UTC) — must include time-of-day"],
    look_back_days: Annotated[int, "bars (intraday) or days (daily) to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.

    IMPORTANT for intraday mode: curr_date MUST include the time-of-day
    in 'YYYY-MM-DD HH:MM' format (UTC). Passing a bare date truncates
    the indicator series at 00:00 UTC of that day and you will miss
    hours of recent bars. Always pass the current datetime shown in the
    system prompt.

    Args:
        symbol (str): Ticker symbol, e.g. AAPL, BTC-USDT
        indicator (str): Single indicator name, e.g. 'rsi', 'macd'. Call once per indicator.
        curr_date (str): 'YYYY-MM-DD' (daily) or 'YYYY-MM-DD HH:MM' UTC (intraday)
        look_back_days (int): Lookback window (bars for intraday, days for daily)
    Returns:
        str: A formatted dataframe containing the technical indicators.
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import get_default_lookback_bars, is_intraday

    interval = get_config().get("trading_interval", "1d")

    # For intraday, override default lookback if caller used the daily default
    if is_intraday(interval) and look_back_days == 30:
        look_back_days = get_default_lookback_bars(interval)

    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days, interval=interval))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)