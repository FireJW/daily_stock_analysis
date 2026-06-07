# Source Methodology

This repository demonstrates a source-grounded market-analysis workflow rather than a financial recommendation product.

## Inputs

- Market data providers: AkShare, Tushare, Baostock, and YFinance.
- Search providers: Tavily and SerpAPI.
- Model providers: Gemini or OpenAI-compatible endpoints.
- User configuration: watchlists, notification channels, and runtime options supplied through environment variables or GitHub Actions secrets.

## Processing Pattern

1. Load a configured watchlist and runtime mode.
2. Collect market quotes, technical context, broad market state, and related news.
3. Prepare bounded prompts for the selected model provider.
4. Generate a structured dashboard, market review, earnings digest, or notification packet.
5. Persist reports/logs and send configured notifications.

## Review Notes

- External provider data can be delayed, partial, or unavailable.
- Search results and LLM outputs should be treated as analysis aids, not facts by default.
- Any investment language in generated output must be reviewed independently.
- Workflow logs can reveal symbols, private watchlists, and notification routing details, so public runs should use sample data.

## Attribution

Original upstream project and codebase: `ZhuLinsen/daily_stock_analysis`.

This FireJW fork is packaged for portfolio review with additional documentation and safety framing.
