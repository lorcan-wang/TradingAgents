from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_current_datetime_label,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_intraday_banner,
    get_language_instruction,
    get_news_timeframe_label,
    get_timeframe_context,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        is_crypto = state["company_of_interest"].upper().replace("-USDT", "-USD").endswith("-USD")

        if is_crypto:
            system_message = (
                get_intraday_banner()
                + "You are a researcher tasked with analyzing fundamental information about a cryptocurrency asset. "
                "Please write a comprehensive report including: market capitalization, circulating supply vs total/max supply, "
                "price performance (24h/7d/30d/1y changes), all-time high/low analysis, developer activity (GitHub commits, forks, stars), "
                "community metrics (social followers, sentiment), and overall market positioning. "
                "Use `get_fundamentals` for market data overview, and `get_balance_sheet`/`get_cashflow`/`get_income_statement` for detailed developer, community, and on-chain metrics. "
                "Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " Make sure to append a Markdown table at the end of the report to organize key points."
                + get_language_instruction(),
            )
        else:
            timeframe_label = get_news_timeframe_label()
            system_message = (
                get_intraday_banner()
                + f"You are a researcher tasked with analyzing fundamental information over the {timeframe_label} about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
                + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
                + get_timeframe_context()
                + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
