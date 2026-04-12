from datetime import datetime
from .alpha_vantage_common import _make_api_request, _filter_csv_by_date_range

def get_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> str:
    """
    Returns raw OHLCV values filtered to the specified date range.

    For daily interval: returns daily adjusted time series.
    For intraday intervals: returns intraday time series.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        interval: Trading interval (1m, 5m, 15m, 30m, 1h, 1d)

    Returns:
        CSV string containing the time series data filtered to the date range.
    """
    from tradingagents.interval_utils import is_intraday, get_av_function, get_av_interval

    # Parse date portion to determine the range
    start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 data points
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    if is_intraday(interval):
        params["interval"] = get_av_interval(interval)

    function_name = get_av_function(interval)
    response = _make_api_request(function_name, params)

    return _filter_csv_by_date_range(response, start_date[:10], end_date[:10])