"""CoinGecko API data fetching for cryptocurrency fundamentals."""

import time
import logging
import threading
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"

# Global rate limiter: max 10 requests per 60 seconds (free tier safe)
_rate_lock = threading.Lock()
_request_timestamps: list[float] = []
_MAX_REQUESTS_PER_MINUTE = 10


def _wait_for_rate_limit():
    """Block until we can safely make a CoinGecko request without hitting rate limits."""
    with _rate_lock:
        now = time.time()
        # Remove timestamps older than 60 seconds
        _request_timestamps[:] = [t for t in _request_timestamps if now - t < 60]
        if len(_request_timestamps) >= _MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - _request_timestamps[0]) + 0.5
            if wait_time > 0:
                logger.info(f"CoinGecko rate limiter: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        _request_timestamps.append(time.time())

# Common crypto ticker to CoinGecko ID mapping
_TICKER_TO_ID = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "DOT": "polkadot", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "LINK": "chainlink", "UNI": "uniswap",
    "ATOM": "cosmos", "LTC": "litecoin", "ETC": "ethereum-classic",
    "NEAR": "near", "APT": "aptos", "ARB": "arbitrum",
    "OP": "optimism", "SUI": "sui", "SEI": "sei-network",
    "TRX": "tron", "SHIB": "shiba-inu", "PEPE": "pepe",
    "FIL": "filecoin", "ICP": "internet-computer",
}


def _parse_crypto_symbol(ticker: str) -> tuple[str, str]:
    """Parse ticker like 'BTC-USD' or 'BTC-USDT' into (symbol='BTC', coingecko_id='bitcoin').

    IMPORTANT: ``-USDT`` must be stripped before ``-USD``, otherwise
    ``BTC-USDT`` becomes ``BTCT`` (the ``-USD`` substring inside ``-USDT``
    matches first).
    """
    symbol = ticker.upper().replace("-USDT", "").replace("-USD", "").strip()
    cg_id = _TICKER_TO_ID.get(symbol)
    if not cg_id:
        # Fallback: use lowercase symbol as id (works for many coins)
        cg_id = symbol.lower()
    return symbol, cg_id


def _cg_request(endpoint: str, params: dict = None, max_retries: int = 3) -> dict:
    """Make a CoinGecko API request with global rate limiting and retry."""
    _wait_for_rate_limit()
    url = f"{_BASE_URL}/{endpoint}"
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"CoinGecko rate limited, retrying in {delay}s")
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            raise
    return {}


def get_crypto_fundamentals(ticker: str, *args, **kwargs) -> str:
    """Get cryptocurrency market fundamentals from CoinGecko.

    Replaces get_fundamentals for crypto assets. Returns market cap,
    supply, price changes, ATH, and volume data.
    """
    symbol, cg_id = _parse_crypto_symbol(ticker)

    try:
        data = _cg_request("coins/markets", {
            "vs_currency": "usd",
            "ids": cg_id,
            "price_change_percentage": "24h,7d,14d,30d,200d,1y",
            "sparkline": "false",
        })

        if not data:
            return f"No fundamentals data found for cryptocurrency '{ticker}'"

        coin = data[0]

        def fmt_num(val, prefix="", suffix=""):
            if val is None:
                return "N/A"
            if isinstance(val, float) and abs(val) >= 1e9:
                return f"{prefix}{val/1e9:.2f}B{suffix}"
            if isinstance(val, float) and abs(val) >= 1e6:
                return f"{prefix}{val/1e6:.2f}M{suffix}"
            return f"{prefix}{val:,.2f}{suffix}" if isinstance(val, float) else f"{prefix}{val:,}{suffix}"

        def fmt_pct(val):
            if val is None:
                return "N/A"
            return f"{val:+.2f}%"

        lines = [
            f"# Cryptocurrency Fundamentals for {coin.get('name', symbol)} ({symbol})",
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Current Price: {fmt_num(coin.get('current_price'), '$')}",
            f"Market Cap: {fmt_num(coin.get('market_cap'), '$')}",
            f"Market Cap Rank: #{coin.get('market_cap_rank', 'N/A')}",
            f"24h Trading Volume: {fmt_num(coin.get('total_volume'), '$')}",
            f"Circulating Supply: {fmt_num(coin.get('circulating_supply'))}",
            f"Total Supply: {fmt_num(coin.get('total_supply'))}",
            f"Max Supply: {fmt_num(coin.get('max_supply')) if coin.get('max_supply') else 'Unlimited'}",
            "",
            "## Price Changes",
            f"24h Change: {fmt_pct(coin.get('price_change_percentage_24h'))}",
            f"7d Change: {fmt_pct(coin.get('price_change_percentage_7d_in_currency'))}",
            f"14d Change: {fmt_pct(coin.get('price_change_percentage_14d_in_currency'))}",
            f"30d Change: {fmt_pct(coin.get('price_change_percentage_30d_in_currency'))}",
            f"200d Change: {fmt_pct(coin.get('price_change_percentage_200d_in_currency'))}",
            f"1y Change: {fmt_pct(coin.get('price_change_percentage_1y_in_currency'))}",
            "",
            "## All-Time High",
            f"ATH Price: {fmt_num(coin.get('ath'), '$')}",
            f"ATH Date: {coin.get('ath_date', 'N/A')}",
            f"ATH Change: {fmt_pct(coin.get('ath_change_percentage'))}",
            "",
            "## All-Time Low",
            f"ATL Price: {fmt_num(coin.get('atl'), '$')}",
            f"ATL Date: {coin.get('atl_date', 'N/A')}",
            f"ATL Change: {fmt_pct(coin.get('atl_change_percentage'))}",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving crypto fundamentals for {ticker}: {str(e)}"


def get_crypto_detail(ticker: str, *args, **kwargs) -> str:
    """Get detailed cryptocurrency data including developer and community metrics.

    Replaces get_balance_sheet/get_cashflow/get_income_statement for crypto.
    Signature is forgiving (``*args, **kwargs``) because it is routed from
    multiple fundamentals tools whose positional args differ (e.g.
    ``get_balance_sheet(ticker, freq, curr_date)`` vs
    ``get_fundamentals(ticker, curr_date)``).
    """
    symbol, cg_id = _parse_crypto_symbol(ticker)

    try:
        data = _cg_request(f"coins/{cg_id}", {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        })

        if not data:
            return f"No detailed data found for cryptocurrency '{ticker}'"

        lines = [
            f"# Cryptocurrency Detail for {data.get('name', symbol)} ({symbol})",
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Description: {(data.get('description', {}).get('en', '') or 'N/A')[:500]}",
            f"Categories: {', '.join(data.get('categories', []) or ['N/A'])}",
            f"Hashing Algorithm: {data.get('hashing_algorithm', 'N/A')}",
            f"Genesis Date: {data.get('genesis_date', 'N/A')}",
        ]

        # Sentiment
        sent_up = data.get("sentiment_votes_up_percentage")
        sent_down = data.get("sentiment_votes_down_percentage")
        if sent_up is not None:
            lines.extend([
                "",
                "## Sentiment",
                f"Positive Votes: {sent_up:.1f}%",
                f"Negative Votes: {sent_down:.1f}%" if sent_down else "",
            ])

        # Developer data
        dev = data.get("developer_data", {})
        if dev:
            lines.extend([
                "",
                "## Developer Activity",
                f"GitHub Forks: {dev.get('forks', 'N/A')}",
                f"GitHub Stars: {dev.get('stars', 'N/A')}",
                f"GitHub Subscribers: {dev.get('subscribers', 'N/A')}",
                f"Total Issues: {dev.get('total_issues', 'N/A')}",
                f"Closed Issues: {dev.get('closed_issues', 'N/A')}",
                f"Pull Requests Merged: {dev.get('pull_requests_merged', 'N/A')}",
                f"Pull Request Contributors: {dev.get('pull_request_contributors', 'N/A')}",
                f"Commit Count (4 weeks): {dev.get('commit_count_4_weeks', 'N/A')}",
            ])

        # Community data
        community = data.get("community_data", {})
        if community:
            lines.extend([
                "",
                "## Community Metrics",
                f"Twitter Followers: {community.get('twitter_followers', 'N/A')}",
                f"Reddit Subscribers: {community.get('reddit_subscribers', 'N/A')}",
                f"Reddit Active Accounts (48h): {community.get('reddit_accounts_active_48h', 'N/A')}",
                f"Telegram Channel Users: {community.get('telegram_channel_user_count', 'N/A')}",
            ])

        # Links
        links = data.get("links", {})
        if links:
            homepage = links.get("homepage", [])
            homepage_url = next((u for u in homepage if u), "N/A")
            lines.extend([
                "",
                "## Links",
                f"Homepage: {homepage_url}",
                f"Whitepaper: {links.get('whitepaper', 'N/A')}",
                f"Subreddit: {links.get('subreddit_url', 'N/A')}",
            ])

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving crypto detail for {ticker}: {str(e)}"


# Per-symbol subreddit mapping for crypto social sentiment.
# Falls back to r/CryptoCurrency (the largest pan-crypto sub) for anything
# not listed here. Reddit JSON is free and requires no auth, just a UA.
_SYMBOL_TO_SUBREDDITS = {
    "BTC": ["Bitcoin", "BitcoinMarkets"],
    "ETH": ["ethereum", "ethfinance"],
    "SOL": ["solana"],
    "BNB": ["binance"],
    "XRP": ["Ripple"],
    "ADA": ["cardano"],
    "DOGE": ["dogecoin"],
    "DOT": ["dot"],
    "AVAX": ["Avax"],
    "MATIC": ["0xPolygon"],
    "LINK": ["Chainlink"],
    "UNI": ["UniSwap"],
    "ATOM": ["cosmosnetwork"],
    "LTC": ["litecoin"],
    "NEAR": ["NEARProtocol"],
    "APT": ["Aptos"],
    "ARB": ["Arbitrum"],
    "OP": ["Optimism"],
    "SUI": ["SuiNetwork"],
    "TRX": ["Tronix"],
    "SHIB": ["SHIBArmy"],
    "PEPE": ["pepecoin"],
    "FIL": ["filecoin"],
    "ICP": ["dfinity"],
    "TON": ["TONcoin"],
    "AAVE": ["Aave_Official"],
}


_REDDIT_HEADERS = {
    "User-Agent": "TradingAgents/0.1 (research; contact: github)"
}


def _fetch_reddit_posts(subreddit: str, limit: int = 10) -> list[dict]:
    """Pull hot posts from a subreddit. Returns [] on any failure."""
    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/hot.json",
            params={"limit": limit},
            headers=_REDDIT_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [c.get("data", {}) for c in data.get("data", {}).get("children", [])]
    except Exception as e:
        logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")
        return []


def get_crypto_status_news(ticker: str, *args, **kwargs) -> str:
    """Ticker-scoped crypto social sentiment via Reddit hot posts.

    Replaces ``get_news`` for crypto tickers. Pulls top posts from the
    coin's primary subreddit(s) — and r/CryptoCurrency as fallback — to
    surface real community discussion that Yahoo Finance's stock-oriented
    ``get_news`` cannot provide for crypto assets.

    Note: previously used CoinGecko's ``coins/{id}/status_updates``
    endpoint, which has been deprecated by CoinGecko (returns 'Incorrect
    path'). Reddit JSON is free, no key required, and gives much richer
    real social signal.
    """
    start_date = args[0] if len(args) >= 1 else kwargs.get("start_date")
    end_date = args[1] if len(args) >= 2 else kwargs.get("end_date")

    try:
        symbol, _cg_id = _parse_crypto_symbol(ticker)
    except Exception:
        return f"No crypto news source available for ticker '{ticker}'"

    subs = _SYMBOL_TO_SUBREDDITS.get(symbol, [])
    # Always include the catch-all crypto sub so we get *something*
    if "CryptoCurrency" not in subs:
        subs = subs + ["CryptoCurrency"]

    header_parts = [f"## {symbol} Social Sentiment (Reddit hot posts)"]
    if start_date or end_date:
        rng = []
        if start_date:
            rng.append(f"from {start_date}")
        if end_date:
            rng.append(f"to {end_date}")
        header_parts.append(" ".join(rng))
    header = " ".join(header_parts) + "\n"

    sections: list[str] = [header]
    total_posts = 0

    for sub in subs:
        posts = _fetch_reddit_posts(sub, limit=10)
        if not posts:
            continue
        # Filter: drop sticky/megathread posts that pollute every sub
        posts = [
            p for p in posts
            if not p.get("stickied") and (p.get("title") or "").strip()
        ]
        if not posts:
            continue

        sections.append(f"\n### r/{sub}\n")
        for p in posts[:8]:
            title = (p.get("title") or "").strip()
            score = p.get("score", 0)
            num_comments = p.get("num_comments", 0)
            created_utc = p.get("created_utc")
            created_str = ""
            if created_utc:
                try:
                    created_str = datetime.utcfromtimestamp(int(created_utc)).strftime("%Y-%m-%d %H:%M UTC")
                except (ValueError, TypeError, OSError):
                    pass
            selftext = (p.get("selftext") or "").strip()

            sections.append(f"- **[{score} ↑ / {num_comments} 💬] {title}**  ({created_str})")
            if selftext:
                snippet = selftext[:300].replace("\n", " ")
                sections.append(f"  > {snippet}")
            total_posts += 1

    if total_posts == 0:
        return (
            header
            + "\nNo Reddit posts retrieved (all subreddit fetches failed). "
            "For broader crypto market sentiment, call `get_global_news` "
            "which routes to the Fear & Greed index for crypto tickers."
        )

    sections.append(
        f"\n---\nRetrieved {total_posts} posts across {len([s for s in subs])} "
        "subreddit(s). Higher score / comment count signals stronger community "
        "interest. Combine with `get_global_news` (Fear & Greed index) for macro "
        "sentiment context."
    )
    return "\n".join(sections)
