from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_current_datetime_label,
    get_indicators,
    get_intraday_banner,
    get_language_instruction,
    get_stock_data,
    get_timeframe_context,
)
from tradingagents.dataflows.config import get_config
from tradingagents.interval_utils import is_intraday


_DAILY_INDICATOR_BLOCK = """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""


_INTRADAY_INDICATOR_BLOCK = """You are an intraday trading assistant. Your role is to select the **most relevant short-period indicators** from the list below to analyze price action on the configured intraday bar size. Choose up to **8 indicators** that give complementary, session-level signals. Each bar in the data represents ONE intraday interval (e.g. a 5-minute bar), so all indicator periods below are measured in BARS, not days.

Short-Period Moving Averages (trend on the intraday bar):
- close_10_ema: 10-bar EMA. Usage: Fast intraday trend proxy — track near-term direction and momentum shifts within the session. Tips: Very responsive; pair with a slower EMA for crossover signals.
- close_50_sma: 50-bar SMA. Usage: Intraday "slower" trend line over the last ~50 bars of the current interval. Acts as dynamic intraday support/resistance. Tips: On small intervals this covers only a few hours — treat it as the session trend anchor, NOT a multi-day trend.
- (DO NOT select close_200_sma — 200 bars of intraday data is irrelevant for a session-level decision.)

MACD (momentum on the intraday bar):
- macd: MACD line computed from intraday EMAs. Usage: Spot intraday momentum shifts and divergences on the configured bar. Tips: On 1m/5m bars MACD flips often — confirm with price action.
- macds: MACD signal line. Usage: Intraday crossover triggers. Tips: Combine with volume to filter noise.
- macdh: MACD histogram. Usage: Visualize intraday momentum strength. Tips: Watch for expanding/contracting bars as session momentum accelerates or fades.

Momentum:
- rsi: RSI on the intraday bar. Usage: Overbought/oversold within the current session (70/30). Tips: In strong intraday trends RSI pins at extremes — don't fade it blindly.

Volatility (critical for intraday sizing):
- boll: Bollinger middle (20-bar SMA on the intraday bar). Usage: Intraday mean-reversion anchor.
- boll_ub: Upper band. Usage: Intraday breakout / overextension level.
- boll_lb: Lower band. Usage: Intraday support / oversold bounce level.
- atr: ATR on the intraday bar. Usage: Measure intraday volatility per bar — critical for setting tight stop-loss levels and position sizes for a session trade. Tips: Use ATR to size stops in price units, not percentages.

Volume (essential intraday):
- vwma: Volume-weighted moving average. Usage: Confirm intraday moves against real participation. Tips: Divergence between price and VWMA on intraday bars often signals weak breakouts.

Guidance:
- Prioritize momentum + volatility + volume indicators. Intraday price action is dominated by short-term momentum and liquidity, not long-term trend.
- Avoid redundancy (do not select both rsi and stochrsi).
- Call get_stock_data FIRST to pull the intraday OHLCV, then call get_indicators with the chosen indicator names. The data will already be in intraday bar resolution — do not request daily data.
- Your report must describe the trend, momentum, volatility, and volume character of the CURRENT intraday session on the configured bar size. Reference specific timestamps (e.g. "at 10:30 the RSI crossed 70"). Do NOT discuss multi-day trends, weekly patterns, or long-term regimes.
- Conclude with a session-level actionable view (lean long / lean short / stand aside) with supporting evidence from the indicators."""


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        interval = get_config().get("trading_interval", "1d")
        indicator_block = _INTRADAY_INDICATOR_BLOCK if is_intraday(interval) else _DAILY_INDICATOR_BLOCK

        system_message = (
            get_intraday_banner()
            + indicator_block
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_timeframe_context()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the {datetime_label} is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(datetime_label=get_current_datetime_label())
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
