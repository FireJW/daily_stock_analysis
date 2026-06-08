# Daily Stock Analysis

Daily Stock Analysis is a Python workflow for running scheduled A-share market reviews, watchlist analysis, news enrichment, and multi-channel notification summaries.

This repository is packaged as a public portfolio version. It keeps the automation structure visible, but credentials, local databases, logs, generated reports, and private runtime artifacts are intentionally excluded.

## What It Does

- Builds a daily stock-watchlist review from configurable tickers.
- Pulls market data from public and optional provider APIs.
- Adds AI-assisted analysis through user-provided model credentials.
- Sends summaries through optional channels such as WeChat Work, Feishu, Telegram, email, or custom webhooks.
- Supports local runs, Docker runs, and GitHub Actions scheduling.

## Repository Layout

```text
main.py                  CLI entrypoint for daily analysis runs
config.py                Environment-driven configuration loader
analyzer.py              Analysis orchestration and AI prompt flow
llm_client.py            Gemini and OpenAI-compatible provider adapter
notification.py          Notification rendering and delivery helpers
storage.py               SQLite persistence helpers
data_provider/           Market data provider adapters
sources/                 Public demo assets
.github/workflows/       CI and scheduled workflow templates
```

## Quick Start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python test_env.py
python main.py --help
```

Fill `.env` with your own API keys and notification endpoints before running a live analysis. The repository does not include working credentials.

## Configuration

Common environment variables:

| Variable | Purpose | Required |
| --- | --- | --- |
| `STOCK_LIST` | Comma-separated watchlist, such as `600519,300750,002594` | Yes |
| `GEMINI_API_KEY` | Google AI Studio key for AI analysis | One AI key required |
| `OPENAI_API_KEY` | OpenAI-compatible fallback key | Optional |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | Optional |
| `TUSHARE_TOKEN` | Optional Tushare Pro token | Optional |
| `TAVILY_API_KEYS` | Optional news search keys | Optional |
| `WECHAT_WEBHOOK_URL` | Optional WeChat Work webhook | Optional |
| `FEISHU_WEBHOOK_URL` | Optional Feishu webhook | Optional |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional Telegram channel | Optional |
| `EMAIL_SENDER` / `EMAIL_PASSWORD` / `EMAIL_RECEIVERS` | Optional email delivery | Optional |

Do not commit `.env`, databases, logs, or generated reports.

## GitHub Actions

The workflow templates under `.github/workflows/` are intended as examples. To use them in your own fork:

1. Add the required secrets in GitHub Actions settings.
2. Review the scheduled run time and stock list.
3. Run the workflow manually once before enabling a schedule.

No repository secret is included in this public package.

## Public Safety

This public version excludes:

- `.env` and local credential files
- SQLite databases and generated reports
- runtime logs
- private notification endpoints
- local machine paths

The diagnostic script reports whether credentials are configured, but it does not print token prefixes or secret values.

## Status

This is a legacy automation project packaged for public review. For a cleaner current stock-research workflow kit, see [Stock Analysis Plus](https://github.com/FireJW/stock-analysis-plus).

## License

MIT License.
