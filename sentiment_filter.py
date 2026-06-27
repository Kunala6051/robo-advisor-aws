"""
sentiment_filter.py
AI-Powered Robo Advisor — Amazon Comprehend Sentiment Analysis Module
Build AI with AWS | June 2026
"""

import boto3

comprehend = boto3.client("comprehend", region_name="us-east-1")

# Sample financial headlines (production: replace with live NewsAPI / S3 feed)
SAMPLE_HEADLINES = {
    "AAPL":  "Apple reports record iPhone sales, beats Q2 estimates by wide margin",
    "MSFT":  "Microsoft Azure growth slows amid cloud spending cuts across enterprises",
    "NVDA":  "NVIDIA GPU demand surges driven by explosive AI training workloads",
    "AMZN":  "Amazon faces major antitrust probe in European Union over marketplace practices",
    "GOOGL": "Alphabet AI Search drives significant ad market share gains in Q2 2026",
    "META":  "Meta reports strong ad revenue recovery, raises full-year guidance",
    "JPM":   "JP Morgan beats earnings estimates, raises 2026 guidance amid strong lending",
    "V":     "Visa reports record cross-border transaction volume, raises dividend",
    "UNH":   "UnitedHealth raises full-year profit forecast on strong membership growth",
    "TSLA":  "Tesla faces significant production delays at Gigafactory Berlin",
    "AVGO":  "Broadcom AI chip orders surge as hyperscaler demand accelerates",
    "LLY":   "Eli Lilly obesity drug sales exceed analyst expectations by 40 percent",
    "HD":    "Home Depot reports mixed results as housing market slowdown continues",
    "MRK":   "Merck cancer drug trial shows strong Phase 3 efficacy results",
    "ABBV":  "AbbVie faces biosimilar competition headwinds for Humira franchise",
    "PG":    "Procter and Gamble raises prices successfully, maintains market share",
    "JNJ":   "Johnson and Johnson settles talc litigation, clears major legal overhang",
    "XOM":   "ExxonMobil beats earnings on higher oil prices and refinery margins",
    "MA":    "Mastercard reports strong consumer spending growth across all regions",
    "BAC":   "Bank of America benefits from higher interest rates, beats Q2 estimates",
    "CVX":   "Chevron increases buyback program amid strong cash generation",
    "COST":  "Costco membership renewals hit record high, same-store sales rise",
    "PEP":   "PepsiCo lowers guidance as volume declines pressure beverage segment",
    "TMO":   "Thermo Fisher wins major biopharma contract, lifts annual outlook",
    "ADBE":  "Adobe beats earnings on AI product suite adoption across creative cloud",
}


def get_sentiment_score(ticker: str) -> tuple:
    """
    Call Amazon Comprehend DetectSentiment on the stock's headline.

    Returns:
        (sentiment_label, net_score)
        net_score = Positive confidence - Negative confidence  (range: -1 to +1)
    """
    headline = SAMPLE_HEADLINES.get(ticker, f"{ticker} stock performance in line with market")

    try:
        response  = comprehend.detect_sentiment(Text=headline, LanguageCode="en")
        sentiment = response["Sentiment"]           # POSITIVE / NEGATIVE / NEUTRAL / MIXED
        scores    = response["SentimentScore"]
        net_score = round(scores["Positive"] - scores["Negative"], 4)
        return sentiment, net_score

    except Exception as e:
        print(f"[Comprehend] Error for {ticker}: {e}")
        return "NEUTRAL", 0.0


def filter_by_sentiment(buy_signals: list, threshold: float = -0.10) -> list:
    """
    Remove stocks with net-negative sentiment score below threshold.
    Sort remaining stocks by net sentiment score (best sentiment first).

    Args:
        buy_signals: Ordered list of ticker strings from portfolio_optimizer
        threshold:   Minimum acceptable net score (default -0.10)

    Returns:
        Filtered and re-sorted list of tickers
    """
    results = []
    filtered_out = []

    for ticker in buy_signals:
        sentiment, net_score = get_sentiment_score(ticker)
        if net_score >= threshold:
            results.append((ticker, net_score, sentiment))
        else:
            filtered_out.append((ticker, net_score, sentiment))
            print(f"[Filter] Removed {ticker} — sentiment: {sentiment}, score: {net_score:.4f}")

    # Sort by net sentiment score descending (most positive news first)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"[Sentiment] Kept {len(results)}, removed {len(filtered_out)} stocks")
    return [t for t, _, _ in results]


def get_sentiment_report(tickers: list) -> list:
    """
    Generate a full sentiment report for a list of tickers.
    Used for logging to DynamoDB and debugging.
    """
    report = []
    for ticker in tickers:
        sentiment, net_score = get_sentiment_score(ticker)
        headline = SAMPLE_HEADLINES.get(ticker, "No headline available")
        report.append({
            "ticker":    ticker,
            "headline":  headline[:80] + "..." if len(headline) > 80 else headline,
            "sentiment": sentiment,
            "net_score": net_score,
        })
    return report
