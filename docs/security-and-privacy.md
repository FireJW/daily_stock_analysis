# Security and Privacy

This project is configured through environment variables and GitHub Actions secrets. Treat all runtime credentials and watchlists as sensitive.

## Do Not Commit

- API keys for Gemini, OpenAI-compatible providers, Tavily, SerpAPI, or Tushare.
- Notification webhooks for WeCom, Feishu, Telegram, email, Slack, Discord, Bark, DingTalk, or custom services.
- Bot tokens, chat IDs, app secrets, folder tokens, cookies, sessions, or private account identifiers.
- Private watchlists, client names, brokerage information, or portfolio holdings.
- Raw workflow logs that include secrets, private symbols, private endpoints, or account-specific notification payloads.

## Public Demo Guidance

- Use placeholder values from `.env.example`.
- Use a small sample watchlist when demonstrating the workflow.
- Disable or mock notification channels for screenshots and public logs.
- Review generated reports before publishing them because LLM output can include unsupported financial claims.

## GitHub Actions

Use repository secrets for credentials and repository variables for non-sensitive defaults. Avoid printing secrets, webhook URLs, or full notification payloads in workflow logs.

## Financial Disclaimer

This repository is for learning and workflow evaluation. It is not investment advice, trading instruction, or financial planning guidance.
