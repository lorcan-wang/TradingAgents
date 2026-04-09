"""Crypto market sentiment from Alternative.me (Fear & Greed Index) and CoinGecko trending."""

import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


def get_fear_greed_index(ticker: str = None, curr_date: str = None) -> str:
    """Get the Crypto Fear & Greed Index from Alternative.me.

    Free, no API key required. Provides overall market sentiment (0-100).
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 30, "date_format": "us"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        entries = data.get("data", [])
        if not entries:
            return "No Fear & Greed Index data available"

        latest = entries[0]
        value = int(latest.get("value", 0))
        classification = latest.get("value_classification", "N/A")

        lines = [
            "# Crypto Fear & Greed Index",
            f"# Source: Alternative.me",
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Current Value: {value}/100",
            f"Classification: {classification}",
            "",
            "## Interpretation",
        ]

        if value <= 25:
            lines.append("Extreme Fear — investors are very worried. Historically a buying opportunity.")
        elif value <= 45:
            lines.append("Fear — market sentiment is negative. Caution prevails.")
        elif value <= 55:
            lines.append("Neutral — market is balanced between fear and greed.")
        elif value <= 75:
            lines.append("Greed — investors are getting greedy. Market may be overheating.")
        else:
            lines.append("Extreme Greed — market is very greedy. Historically signals a correction risk.")

        lines.extend(["", "## Recent History (Last 7 Days)"])
        for entry in entries[:7]:
            ts = entry.get("timestamp", "")
            try:
                dt = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
            except (ValueError, TypeError, OSError):
                dt = "N/A"
            val = entry.get("value", "N/A")
            cls = entry.get("value_classification", "")
            lines.append(f"  {dt}: {val} ({cls})")

        # 30-day trend summary
        if len(entries) >= 7:
            values = [int(e.get("value", 0)) for e in entries[:30] if e.get("value")]
            if values:
                avg = sum(values) / len(values)
                lines.extend([
                    "",
                    f"## 30-Day Summary",
                    f"Average: {avg:.1f}",
                    f"Highest: {max(values)}",
                    f"Lowest: {min(values)}",
                ])

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving Fear & Greed Index: {str(e)}"


def get_crypto_trending(ticker: str = None, curr_date: str = None) -> str:
    """Get trending cryptocurrencies from CoinGecko.

    Free, no API key required. Shows what the market is most interested in.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        coins = data.get("coins", [])
        if not coins:
            return "No trending data available from CoinGecko"

        lines = [
            "# Trending Cryptocurrencies",
            f"# Source: CoinGecko",
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for i, coin_wrapper in enumerate(coins[:15], 1):
            coin = coin_wrapper.get("item", {})
            name = coin.get("name", "N/A")
            symbol = coin.get("symbol", "N/A")
            rank = coin.get("market_cap_rank", "N/A")
            price_btc = coin.get("price_btc")
            price_str = f"{price_btc:.8f} BTC" if price_btc else "N/A"
            score = coin.get("score", "N/A")

            lines.append(f"{i}. **{name}** ({symbol}) — Rank #{rank}, {price_str}")

        # Also show trending categories if available
        categories = data.get("categories", [])
        if categories:
            lines.extend(["", "## Trending Categories"])
            for cat in categories[:5]:
                cat_data = cat.get("item", cat) if isinstance(cat, dict) else {}
                if isinstance(cat_data, dict):
                    lines.append(f"- {cat_data.get('name', str(cat))}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving trending data: {str(e)}"
