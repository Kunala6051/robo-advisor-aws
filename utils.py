"""
utils.py
AI-Powered Robo Advisor — Utility Functions (DynamoDB Logging + Response Formatting)
Build AI with AWS | June 2026
"""

import boto3
import json
import hashlib
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
TABLE_NAME = "RoboAdvisorPortfolios"


def log_to_dynamo(
    risk: str,
    amount: float,
    horizon: int,
    portfolio: dict,
    sentiment_report: list = None,
) -> bool:
    """
    Persist portfolio request + results to DynamoDB for audit and future retraining.

    Args:
        risk:             Risk level string
        amount:           Investment amount
        horizon:          Time horizon in trading days
        portfolio:        Portfolio dict from portfolio_optimizer
        sentiment_report: Optional list of sentiment dicts from sentiment_filter

    Returns:
        True if successful, False otherwise
    """
    try:
        table = dynamodb.Table(TABLE_NAME)

        # Stable session ID (hash of core params so same request maps to same ID)
        session_id = "session-" + hashlib.md5(
            f"{risk}{amount}{horizon}".encode()
        ).hexdigest()[:8]

        item = {
            "UserID":            session_id,
            "Timestamp":         datetime.now(timezone.utc).isoformat(),
            "RiskLevel":         risk.lower(),
            "Amount":            str(amount),
            "Horizon":           str(horizon),
            "NumStocks":         str(portfolio["n"]),
            "Stocks":            json.dumps(portfolio["stocks"]),
            "WeightPerStock":    str(portfolio["weight"]),
            "ModelVersion":      portfolio.get("model_version", "RF-v1.0"),
            "SentimentFiltered": "true" if sentiment_report else "false",
        }

        if sentiment_report:
            item["SentimentReport"] = json.dumps(sentiment_report)

        table.put_item(Item=item)
        print(f"[DynamoDB] Logged portfolio for {session_id}")
        return True

    except Exception as e:
        print(f"[DynamoDB] Error logging portfolio: {e}")
        return False


def format_lex_response(intent_name: str, message: str, state: str = "Fulfilled") -> dict:
    """
    Format a Lex V2 compatible response dict.

    Args:
        intent_name: The Lex intent name (e.g. 'GetPortfolio')
        message:     The response text to show the user
        state:       Intent fulfillment state ('Fulfilled' | 'Failed')

    Returns:
        Lex V2 response dict
    """
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {
                "name":  intent_name,
                "state": state,
            },
        },
        "messages": [
            {
                "contentType": "PlainText",
                "content":     message,
            }
        ],
    }


def safe_get_slot(slots: dict, slot_name: str, default=None):
    """
    Safely extract a slot value from Lex V2 event slots dict.

    Args:
        slots:     Slots dict from event['sessionState']['intent']['slots']
        slot_name: Name of the slot to extract
        default:   Default value if slot is missing or unresolved

    Returns:
        Interpreted slot value string, or default
    """
    try:
        return slots[slot_name]["value"]["interpretedValue"]
    except (KeyError, TypeError):
        return default
