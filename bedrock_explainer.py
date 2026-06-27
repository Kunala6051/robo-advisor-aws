"""
bedrock_explainer.py
AI-Powered Robo Advisor — Amazon Bedrock Generative AI Explanation Module
Build AI with AWS | June 2026
Uses Claude claude-sonnet-4-6 via Amazon Bedrock to generate portfolio explanations.
"""

import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "anthropic.claude-sonnet-4-6"


def generate_explanation(portfolio: dict) -> str:
    """
    Generate a plain-English portfolio explanation using Amazon Bedrock (Claude claude-sonnet-4-6).

    Args:
        portfolio: dict from portfolio_optimizer containing stocks, risk, amount, etc.

    Returns:
        Natural-language explanation string (3-4 sentences)
    """
    top5 = ", ".join(portfolio["stocks"][:5])
    remaining = len(portfolio["stocks"]) - 5

    prompt = f"""You are a professional AI financial advisor assistant working at a top-tier wealth management firm.

A client has requested a portfolio with the following parameters:
- Risk level: {portfolio['risk'].upper()}
- Investment amount: ${portfolio['amount']:,.0f}
- Time horizon: {portfolio['horizon']} trading days
- Selected stocks: {top5} (and {remaining} more)
- Allocation per stock: ${portfolio['weight']:,.2f} (equal-weight)
- ML model used: Random Forest Classifier (~55% directional accuracy on 2022-2024 test data)
- Sentiment filter: Amazon Comprehend analysed news headlines; stocks with net-negative sentiment were removed

In exactly 3 sentences, provide:
1. Why this allocation suits the client's stated risk profile and time horizon
2. What value the Amazon Comprehend sentiment filter added to the stock selection
3. The key risks the client should be aware of (model accuracy limits, equal-weight assumptions, no real-time execution)

Keep the tone professional, concise, and honest. Do not fabricate specific return figures."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 350,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    })

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    except Exception as e:
        print(f"[Bedrock] Error generating explanation: {e}")
        return (
            f"Your {portfolio['risk']}-risk portfolio allocates ${portfolio['weight']:,.2f} equally "
            f"across {portfolio['n']} ML-selected stocks over a {portfolio['horizon']}-day horizon. "
            "Stocks were filtered by Amazon Comprehend sentiment analysis to remove negative-news exposure. "
            "Note: model accuracy is ~55%; this is algorithmic guidance, not guaranteed financial advice."
        )


def generate_performance_summary() -> str:
    """
    Generate a natural-language performance summary using Bedrock.
    Used for the GetPerformance Lex intent.
    """
    prompt = """You are a financial advisor AI. Summarise the following trading strategy results in 2 sentences:
- Model: Random Forest Classifier trained on top 50 S&P 500 stocks (2019-2022), tested 2022-2024
- ML Strategy cumulative return: ~18% (2022-2024)
- Buy & Hold benchmark return: ~12% (2022-2024)
- Alpha generated: ~6%
- Model test accuracy: ~55.1%
Be concise and professional."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}],
    })

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        print(f"[Bedrock] Performance summary error: {e}")
        return (
            "ML Strategy: ~18% return (2022-2024) | "
            "Buy & Hold Benchmark: ~12% | Alpha: ~6% | "
            "Model: Random Forest | Accuracy: ~55%"
        )
