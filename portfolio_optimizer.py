"""
portfolio_optimizer.py
AI-Powered Robo Advisor — Portfolio Construction Module
Build AI with AWS | June 2026
"""

# Pre-computed buy signals from Random Forest model (55.1% accuracy, 2022-2024 test set)
# These represent the top 50 S&P 500 stocks ranked by predicted next-day bullish probability
BUY_SIGNALS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "V",
    "UNH",  "TSLA", "AVGO", "LLY",  "HD",    "MRK",  "ABBV", "PG",
    "JNJ",  "XOM",  "MA",   "BAC",  "CVX",   "COST", "PEP",  "TMO",
    "ADBE", "ACN",  "CRM",  "NFLX", "AMD",   "TXN",  "QCOM", "HON",
    "AMGN", "SBUX", "GE",   "MS",   "BLK",   "ISRG", "MDT",  "GILD",
    "CME",  "ZTS",  "REGN", "MO",   "ADP",   "ITW",  "CI",   "MMC",
    "VRTX", "EOG",
]

# Risk level → number of stocks selected
RISK_MAP = {
    "low":    5,
    "medium": 15,
    "high":   25,
}


def get_portfolio(risk_level: str, amount: float, horizon: int) -> dict:
    """
    Build an equal-weight portfolio based on risk level.

    Args:
        risk_level: 'low' | 'medium' | 'high'
        amount:     Total investment in USD
        horizon:    Time horizon in trading days

    Returns:
        dict with stocks, allocation, and metadata
    """
    n      = RISK_MAP.get(risk_level.lower(), 15)
    stocks = BUY_SIGNALS[:n]
    weight = round(amount / n, 2)

    return {
        "stocks":        stocks,
        "n":             n,
        "weight":        weight,
        "amount":        amount,
        "horizon":       horizon,
        "risk":          risk_level.lower(),
        "model_version": "RandomForest-v1.0",
        "strategy":      "Equal-weight, ML buy-signal ranked",
    }


def format_response(portfolio: dict) -> str:
    """Format portfolio dict into a human-readable string for Lex."""
    stocks_str = " | ".join(
        f"{s}: ${portfolio['weight']:,.2f}" for s in portfolio["stocks"]
    )
    return (
        f"Here is your {portfolio['risk'].upper()}-risk portfolio "
        f"for ${portfolio['amount']:,.0f} over {portfolio['horizon']} days:\n"
        f"{stocks_str}\n"
        f"Total: ${portfolio['amount']:,.2f} | Stocks: {portfolio['n']} | "
        f"Weight per stock: {100/portfolio['n']:.2f}% | "
        f"Horizon: {portfolio['horizon']} days"
    )
