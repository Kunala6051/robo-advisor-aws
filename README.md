# AI-Powered Robo Advisor for Wealth Management
**Course:** Build AI with AWS | **Weeks:** 3–5 | **June 2026**

---

## Overview

An end-to-end algorithmic trading bot that uses Machine Learning to generate buy/sell signals for the top 50 S&P 500 companies. Users interact via an AWS Lex chatbot in natural language; AWS Lambda executes portfolio optimization and integrates two AWS AI services — Amazon Comprehend for sentiment analysis and Amazon Bedrock for Generative AI explanations.

---

## Architecture

```
User (Web / Mobile)
        │
        ▼ Natural Language Request
  Amazon Lex V2 (RoboAdvisorBot)
        │
        ▼ Intent + Slots (RiskLevel, Amount, Horizon)
   AWS Lambda (RoboAdvisorFunction — Python 3.11)
        │
        ├─[1]─ portfolio_optimizer.py   →  Random Forest buy signals
        ├─[2]─ sentiment_filter.py      →  Amazon Comprehend (news sentiment)
        ├─[3]─ bedrock_explainer.py     →  Amazon Bedrock / Claude claude-sonnet-4-6
        └─[4]─ utils.py                 →  DynamoDB logging
        │
        ▼
  Amazon DynamoDB (RoboAdvisorPortfolios)
        │
        ▼
  Portfolio Recommendation + AI Explanation → User
```

---

## Repository Structure

```
robo-advisor-aws/
├── README.md
├── requirements.txt
├── notebooks/
│   └── trading_bot.ipynb          # Full ML pipeline (Weeks 3–5)
├── lambda/
│   ├── lambda_function.py         # Main Lambda handler
│   ├── portfolio_optimizer.py     # Risk-level portfolio logic
│   ├── sentiment_filter.py        # Amazon Comprehend integration
│   ├── bedrock_explainer.py       # Amazon Bedrock (Claude) integration
│   └── utils.py                   # DynamoDB + response utilities
├── api/
│   └── api_gateway_config.json    # API Gateway Swagger definition
├── database/
│   └── dynamo_schema.json         # DynamoDB table schema + deployment steps
├── plots/
│   └── cumulative_returns.png     # ML Strategy vs Buy & Hold chart
└── docs/
    ├── Week3_Report.pdf
    ├── Week4_Report.pdf
    └── Week5_Report.pdf
```

---

## ML Models

| Model | Features | Test Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 1/5/20-day returns, MA ratios, volatility | 52.8% | Baseline, interpretable |
| **Random Forest** ✓ | Same 6 features | **52.6%** | **Selected — Better for Handling non-linear data** |

**Features used:**
- `ret_1d`, `ret_5d`, `ret_20d` — Short/medium/long-term returns
- `ma5_ratio`, `ma20_ratio` — Price relative to 5/20-day moving average
- `vol_20d` — 20-day realized volatility

---

## Backtest Results (2022–2024, Medium Risk)

| Metric | Value |
|---|---|
| ML Strategy Return | ~18% |
| Buy & Hold Benchmark | ~12% |
| Alpha Generated | **~6%** |
| Dataset | Top 50 S&P 500 by market cap |
| Train period | 2019–2022 |
| Test period | 2022–2024 |

---

## AWS Services Used

| Service | Purpose |
|---|---|
| **Amazon Lex V2** | NLU chatbot — 3 intents (GetPortfolio, GetPerformance, Greet) |
| **AWS Lambda** | Serverless backend orchestrating full AI pipeline |
| **Amazon Comprehend** | DetectSentiment on financial news headlines |
| **Amazon Bedrock** | Claude claude-sonnet-4-6 — natural-language portfolio explanations |
| **Amazon DynamoDB** | Audit log of all portfolio requests + sentiment scores |
| **Amazon API Gateway** | REST endpoints for web/mobile frontend |

---

## Chatbot Intents

### GetPortfolio
```
"Give me a medium risk portfolio for $10,000 over 5 days"
"I want to invest $5000 with low risk"
"Build me a high risk portfolio for $20,000 over 10 days"
```
**Slots:** RiskLevel (low/medium/high) | Amount (USD) | Horizon (days)

### GetPerformance
```
"How is the strategy performing?"
"Show me returns"
"What's the alpha?"
```

### Greet
```
"Hello" | "Hi" | "Help" | "What can you do?"
```

---

## Risk Levels

| Level | Stocks Selected | Use Case |
|---|---|---|
| Low | Top 5 | Conservative, capital preservation |
| Medium | Top 15 | Balanced growth |
| High | Top 25 | Aggressive, higher volatility |

All portfolios use **equal-weight allocation**.

---

## Setup & Deployment

### Step 1 — Run Jupyter Notebook
```bash
# Google Colab / SageMaker
pip install yfinance scikit-learn pandas numpy matplotlib
jupyter notebook notebooks/trading_bot.ipynb
# Run all cells top-to-bottom
# Outputs: cumulative_returns.png, model_comparison.png
```

### Step 2 — Deploy Lambda
1. Console → Lambda → Create Function → **RoboAdvisorFunction**
2. Runtime: Python 3.11 | Memory: 256 MB | Timeout: 30s
3. Upload all files from `lambda/` directory
4. Set environment variable: `TABLE_NAME=RoboAdvisorPortfolios`
5. Attach IAM role with DynamoDB + Comprehend + Bedrock permissions

### Step 3 — Create DynamoDB Table
See `database/dynamo_schema.json` for full schema and deployment steps.
```
Table name: RoboAdvisorPortfolios
Partition key: UserID (String)
Sort key: Timestamp (String)
Billing: On-demand
```

### Step 4 — Create Lex Bot
1. Console → Amazon Lex → Create Bot → **RoboAdvisorBot**
2. Add 3 intents: GetPortfolio, GetPerformance, Greet
3. Configure slots for GetPortfolio (RiskLevel, Amount, Horizon)
4. Set Lambda fulfillment → **RoboAdvisorFunction**
5. Build and test the bot

### Step 5 — Deploy API Gateway
See `api/api_gateway_config.json` for full Swagger definition.
```
POST /portfolio  — Generate portfolio
GET  /performance — Backtest metrics
GET  /health     — Health check
Stage: prod
```

---

## IAM Permissions Required

Lambda execution role needs:
```json
{
  "actions": [
    "dynamodb:PutItem",
    "dynamodb:GetItem",
    "dynamodb:Query",
    "comprehend:DetectSentiment",
    "bedrock:InvokeModel",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ]
}
```

---

## Limitations & Future Work

- **Equal-weight allocation** — Future: mean-variance optimization, Sharpe ratio maximization
- **Model accuracy ~55%** — Future: LSTM/Transformer on longer sequences, NLP features
- **End-of-day data** — Future: real-time streaming via AWS Kinesis Data Streams
- **US stocks only** — Future: NSE India, LSE, ETFs, crypto assets
- **No transaction costs** — Future: realistic execution assumptions (slippage, commissions)
- **Static headlines** — Future: live news feed via NewsAPI.org integrated with S3

---

## Technologies

**Python:** pandas, numpy, scikit-learn, yfinance, matplotlib  
**AWS:** Lambda, Lex V2, Comprehend, Bedrock, DynamoDB, API Gateway  
**Dev:** Google Colab / Jupyter Notebook / AWS SageMaker

---

## Performance Summary

```
Random Forest Model  : 52.6% test accuracy (vs 50% random baseline)
ML Strategy          : ~18% cumulative return (2022-2024)
Buy & Hold Benchmark : ~12% cumulative return (2022-2024)
Alpha Generated      : ~6%
Lambda Avg Latency   : ~4.8 seconds (full AI pipeline)
Bedrock Avg Latency  : ~2.1 seconds (Claude claude-sonnet-4-6, 300 tokens)
```

---

*Build AI with AWS | June 2026 | Chitkara University*
