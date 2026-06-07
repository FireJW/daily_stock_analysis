# Daily Stock Analysis

Portfolio fork of an AI-assisted A-share analysis workflow that combines market data, news search, LLM reasoning, and scheduled notification delivery.

This repository is maintained as a public portfolio packaging of the upstream project by `ZhuLinsen/daily_stock_analysis`. The README, documentation, and GitHub profile text here focus on the system design, automation boundaries, and safe evaluation workflow rather than claiming original authorship of the upstream code.

## What It Demonstrates

- Multi-source market data ingestion through AkShare, Tushare, Baostock, and YFinance adapters.
- LLM-backed analysis with Gemini or OpenAI-compatible providers.
- News and event enrichment through Tavily and SerpAPI.
- Scheduled GitHub Actions execution for daily market review, stock watchlist analysis, and earnings checks.
- Multi-channel delivery through WeCom, Feishu, Telegram, email, and custom webhooks.
- Practical operational controls: rate limits, retries, local SQLite storage, logs, and report artifacts.

## Portfolio Context

This fork is useful as a reviewable example of:

- how a market-analysis automation is structured end to end;
- how environment-driven secrets and notification channels are wired into GitHub Actions;
- how AI output can be converted into checklists and decision dashboards;
- how to document financial automation with explicit risk and privacy boundaries.

The project is not investment advice, trading infrastructure, or a managed financial product.

## System Flow

```text
Watchlist / schedule
        |
        v
Market data providers + news search
        |
        v
Feature preparation and market context
        |
        v
LLM analysis prompt and fallback model routing
        |
        v
Decision dashboard, market review, earnings digest
        |
        v
Reports, logs, notification channels
```

## Quick Start

Clone and install dependencies:

```bash
git clone https://github.com/FireJW/daily_stock_analysis.git
cd daily_stock_analysis
pip install -r requirements.txt
```

Create local configuration:

```bash
cp .env.example .env
```

At minimum, configure:

```bash
STOCK_LIST=600519,300750,002594
GEMINI_API_KEY=your_gemini_key
```

Run locally:

```bash
python main.py
python main.py --market-review
python main.py --schedule
```

For GitHub Actions execution, configure repository secrets for model keys, stock lists, and notification channels, then run the `每日股票分析` workflow manually or on schedule.

## Configuration Surface

Core inputs are environment variables:

| Area | Variables |
| --- | --- |
| Watchlist | `STOCK_LIST`, `US_STOCK_LIST` |
| Model access | `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| Search | `TAVILY_API_KEYS`, `SERPAPI_API_KEYS` |
| Market data | `TUSHARE_TOKEN` |
| Notifications | `WECHAT_WEBHOOK_URL`, `FEISHU_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `EMAIL_SENDER`, `EMAIL_PASSWORD`, `CUSTOM_WEBHOOK_URLS` |
| Runtime | `DATABASE_PATH`, `LOG_DIR`, `LOG_LEVEL`, `MAX_WORKERS`, `SCHEDULE_ENABLED`, `SCHEDULE_TIME` |

See `.env.example` for the full template.

## Repository Map

```text
daily_stock_analysis/
├── main.py                 # CLI and workflow entry point
├── analyzer.py             # LLM-assisted stock analysis
├── market_analyzer.py      # Market review pipeline
├── search_service.py       # News and web search enrichment
├── notification.py         # Notification adapters
├── scheduler.py            # Local scheduled execution
├── storage.py              # SQLite-backed storage
├── config.py               # Environment-driven configuration
├── data_provider/          # Market data provider adapters
├── .github/workflows/      # GitHub Actions automation
├── docs/                   # Portfolio and safety documentation
└── sources/                # Demo screenshots and media from the upstream project
```

## Documentation

- [Portfolio overview](https://firejw.github.io/daily_stock_analysis/)
- [Source methodology](docs/source-methodology.md)
- [Security and privacy](docs/security-and-privacy.md)
- [Deployment notes](DEPLOY.md)

## Safety Notes

- This repository is for learning, portfolio review, and workflow evaluation.
- Outputs can be incomplete, stale, or wrong because they depend on external data providers, search APIs, and LLM responses.
- Do not put real API keys, cookies, tokens, account identifiers, or private watchlists in commits, issues, screenshots, or public logs.
- Treat any generated buy, sell, stop-loss, or target-price language as an analysis artifact, not as financial advice.

## Upstream Attribution

Original upstream project and codebase: `ZhuLinsen/daily_stock_analysis`.

This FireJW repository keeps that attribution visible and packages the fork for public review with clearer portfolio positioning, safety notes, and GitHub Pages documentation.

## License

MIT License. See [LICENSE](LICENSE).
