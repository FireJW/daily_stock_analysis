# Demo Runbook

Use this runbook for public demos without exposing personal credentials, private watchlists, or notification endpoints.

## Local Dry Run

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Use a small sample watchlist such as `600519,300750,002594`.
5. Leave notification webhooks blank unless running in a private environment.
6. Run `python main.py --market-review` or `python main.py`.

## GitHub Actions Demo

1. Configure model and search credentials as repository secrets.
2. Configure `STOCK_LIST` with sample symbols.
3. Disable real notification endpoints for public demos.
4. Trigger the daily stock analysis workflow manually.
5. Review reports and logs before sharing screenshots.

## Review Checklist

- No real API keys, webhook URLs, chat IDs, app secrets, cookies, or private endpoints appear in logs.
- Screenshots use sample symbols and do not show account or client information.
- Generated analysis is labeled as workflow output, not investment advice.
- Any upstream attribution remains visible.
