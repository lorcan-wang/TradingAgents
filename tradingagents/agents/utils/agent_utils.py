from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to all agents (analysts, researchers, traders, risk managers).
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_timeframe_context() -> str:
    """Return intraday context string for analyst prompts.

    Returns empty string for daily mode so existing prompts are unchanged.
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import get_prompt_timeframe_context
    interval = get_config().get("trading_interval", "1d")
    return get_prompt_timeframe_context(interval)


def get_news_timeframe_label() -> str:
    """Return 'past 24 hours' for intraday, 'past week' for daily."""
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import is_intraday
    interval = get_config().get("trading_interval", "1d")
    return "past 24 hours" if is_intraday(interval) else "past week"


def get_current_datetime_label() -> str:
    """Return 'current date and time' for intraday, 'current date' for daily.

    Used in prompt templates so intraday mode signals to the LLM that the
    timestamp is a datetime, not just a date.
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import is_intraday
    interval = get_config().get("trading_interval", "1d")
    return "current date and time" if is_intraday(interval) else "current date"


def get_intraday_banner() -> str:
    """Return a strong intraday-mode header to prepend to analyst system messages.

    Empty string when daily — keeps daily prompts byte-identical to the
    pre-intraday version so existing behavior is preserved.
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import is_intraday
    interval = get_config().get("trading_interval", "1d")
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
        f"[INTRADAY TRADING MODE — {label} bars]\n"
        "You are performing INTRADAY trading analysis. The trading horizon is "
        "within the current trading session (minutes to a few hours), NOT days "
        "or weeks. Your entire analysis and BUY/HOLD/SELL recommendation must "
        f"be framed around the {label} timeframe.\n"
        "Focus on: short-term price action and momentum, intraday volume "
        "profile, session-level support/resistance, short-period technical "
        "indicators, and news/events within the current session.\n"
        "DO NOT base the recommendation on multi-day or weekly trends, "
        "long-term fundamentals, or long-period moving averages (e.g. 50/200 "
        "SMA). Those are irrelevant to the current decision.\n"
        "CRITICAL — TOOL CALLING: when you call get_stock_data or "
        "get_indicators, you MUST pass the FULL current datetime "
        "('YYYY-MM-DD HH:MM', interpreted as UTC) — NOT just a date. "
        "The exact datetime is provided in the system prompt below. "
        "Passing only 'YYYY-MM-DD' truncates the data at 00:00 UTC of "
        "that day, causing the 'current price' you report to be hours "
        "stale — this is the single most common intraday mistake.\n\n"
    )


def get_intraday_decision_context() -> str:
    """Return a short intraday-framing block for downstream non-analyst agents.

    The analyst banner (``get_intraday_banner``) is large and tool-call
    focused. Researchers, the trader, risk debators and the portfolio
    manager don't make tool calls — they just need to know that the
    decision they are debating is a session-level intraday trade, not a
    multi-week investment thesis. Otherwise their default prompt language
    ("growth potential", "long-term sustainability", "portfolio
    allocation") drags the final BUY/HOLD/SELL toward a long-term frame
    that is mismatched with the intraday data the analysts produced.

    Returns empty string in daily mode so existing prompts stay
    byte-identical.
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.interval_utils import is_intraday
    interval = get_config().get("trading_interval", "1d")
    if not is_intraday(interval):
        return ""

    labels = {
        "1m": ("1-minute", "the next few minutes"),
        "5m": ("5-minute", "the next 30-60 minutes"),
        "15m": ("15-minute", "the next 1-3 hours"),
        "30m": ("30-minute", "the next 2-4 hours"),
        "1h": ("1-hour", "the next few hours within today's session"),
    }
    bar_label, horizon = labels.get(interval, (interval, "the current session"))
    return (
        f"[INTRADAY DECISION — {bar_label} bars, horizon: {horizon}]\n"
        "This is a SESSION-LEVEL intraday trade, not a multi-day or multi-week "
        "investment thesis. Frame all reasoning around short-term momentum, "
        "intraday volatility, session support/resistance, order-flow signals, "
        "and news/events within the current session. The position will likely "
        "be opened and closed within the same trading session.\n"
        "DO NOT argue from quarterly fundamentals, multi-year growth potential, "
        "long-term competitive moats, or strategic portfolio allocation — those "
        "framings are mismatched with the actual decision horizon. 'Hold' here "
        "means 'do not take a new intraday position right now', NOT 'keep "
        "owning the stock long-term'.\n\n"
    )


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
