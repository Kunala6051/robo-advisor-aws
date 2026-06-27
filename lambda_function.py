"""
lambda_function.py
AI-Powered Robo Advisor — AWS Lambda Main Handler
Build AI with AWS | June 2026

Architecture:
  Amazon Lex V2
       ↓
  AWS Lambda (this file)
       ↓
  [1] portfolio_optimizer.py  — Random Forest buy signals
  [2] sentiment_filter.py     — Amazon Comprehend news sentiment
  [3] bedrock_explainer.py    — Amazon Bedrock (Claude) explanation
  [4] utils.py                — DynamoDB logging + Lex response format

Function Config:
  Name:    RoboAdvisorFunction
  Runtime: Python 3.11
  Memory:  256 MB
  Timeout: 30 seconds
  Trigger: Amazon Lex (RoboAdvisorBot) + API Gateway (prod stage)
"""

from portfolio_optimizer import get_portfolio, format_response
from sentiment_filter    import filter_by_sentiment, get_sentiment_report
from bedrock_explainer   import generate_explanation, generate_performance_summary
from utils               import log_to_dynamo, format_lex_response, safe_get_slot


def lambda_handler(event, context):
    """
    Main Lambda entry point — handles all Lex V2 intents.

    Supported intents:
      GetPortfolio   — Build optimised portfolio for given risk/amount/horizon
      GetPerformance — Return backtest performance metrics
      Greet          — Welcome message + usage instructions
      FallbackIntent — Graceful fallback for unrecognised input
    """
    print(f"[Lambda] Received event: {event}")

    intent_name = event["sessionState"]["intent"]["name"]
    slots       = event["sessionState"]["intent"].get("slots", {})

    # ── Intent: GetPortfolio ──────────────────────────────────────────────────
    if intent_name == "GetPortfolio":
        risk    = safe_get_slot(slots, "RiskLevel", default="medium")
        amount  = float(safe_get_slot(slots, "Amount",    default="10000"))
        horizon = int(float(safe_get_slot(slots, "Horizon", default="5")))

        # Step 1: ML model — Random Forest buy signals
        portfolio = get_portfolio(risk, amount, horizon)

        # Step 2: AWS AI — Amazon Comprehend sentiment filter
        filtered_stocks    = filter_by_sentiment(portfolio["stocks"])
        sentiment_report   = get_sentiment_report(filtered_stocks)
        portfolio["stocks"] = filtered_stocks
        portfolio["n"]      = len(filtered_stocks)
        portfolio["weight"] = round(amount / max(portfolio["n"], 1), 2)

        # Step 3: AWS GenAI — Amazon Bedrock (Claude) explanation
        explanation = generate_explanation(portfolio)

        # Step 4: DynamoDB logging
        log_to_dynamo(risk, amount, horizon, portfolio, sentiment_report)

        # Compose final message
        portfolio_text = format_response(portfolio)
        message = f"{portfolio_text}\n\n--- AI Advisor Note ---\n{explanation}"

    # ── Intent: GetPerformance ────────────────────────────────────────────────
    elif intent_name == "GetPerformance":
        message = generate_performance_summary()

    # ── Intent: Greet / FallbackIntent ───────────────────────────────────────
    else:
        message = (
            "Hello! I am your AI Robo Advisor, powered by Machine Learning and Amazon Bedrock.\n\n"
            "Here is what I can do:\n"
            "  • Portfolio: 'Give me a medium risk portfolio for $10,000 over 5 days'\n"
            "  • Performance: 'How is the strategy performing?'\n"
            "  • Risk levels: low (5 stocks) | medium (15 stocks) | high (25 stocks)\n\n"
            "How can I help you today?"
        )

    print(f"[Lambda] Responding with intent={intent_name}, message_len={len(message)}")
    return format_lex_response(intent_name, message)
