from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date. Daily: 'YYYY-MM-DD'. Intraday: 'YYYY-MM-DD HH:MM' (UTC)"],
    end_date: Annotated[str, "End date. Daily: 'YYYY-MM-DD'. Intraday: 'YYYY-MM-DD HH:MM' (UTC) — use the CURRENT datetime to get the latest bar"],
) -> str:
    """
    Retrieve OHLCV price data for a given ticker symbol.
    Uses the configured core_stock_apis vendor.

    IMPORTANT for intraday mode: end_date MUST include the time-of-day in
    'YYYY-MM-DD HH:MM' format (interpreted as UTC). Passing a bare date
    truncates data to 00:00 UTC of that day and you will miss hours of
    recent bars — the reported "current price" will be stale. Use the
    current datetime shown in the system prompt as end_date.

    Args:
        symbol (str): Ticker symbol, e.g. AAPL, BTC-USDT
        start_date (str): 'YYYY-MM-DD' (daily) or 'YYYY-MM-DD HH:MM' UTC (intraday)
        end_date (str): 'YYYY-MM-DD' (daily) or 'YYYY-MM-DD HH:MM' UTC (intraday)
    Returns:
        str: A formatted dataframe containing the OHLCV data.
    """
    from tradingagents.dataflows.config import get_config
    interval = get_config().get("trading_interval", "1d")
    return route_to_vendor("get_stock_data", symbol, start_date, end_date, interval=interval)
