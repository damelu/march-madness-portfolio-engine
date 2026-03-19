# Tooling And Crawl Strategy

## Tooling decision

The current workspace already exposes Playwright MCP tools, web search, and general shell access. That is enough to start.

Installed locally in this project:

- `@playwright/mcp`
- `wrangler`

These are intentionally minimal:

- `@playwright/mcp` gives a reproducible local MCP browser server for targeted interactive extraction.
- `wrangler` gives the official Cloudflare CLI needed to work cleanly with Browser Rendering and related APIs later.

I did not install a sports-specific MCP server. NCAA data is better collected through direct APIs, package exports, and explicit ETL scripts than through conversational wrappers.

## What to use when

### Use direct HTTP or APIs when

- The source already exposes structured JSON, CSV, or package exports.
- You need deterministic backfills.
- You need repeatable nightly or weekly pulls.

Examples:

- NCAA hidden feeds.
- SportsDataIO.
- Sportradar.
- The Odds API.

### Use Cloudflare Browser Rendering `/crawl` when

- A site allows crawling.
- You want site-wide discovery from one starting URL.
- You need Markdown or HTML captures of staff directories, team bios, media guides, or press archives.
- You want incremental refresh with `maxAge` and `modifiedSince`.

Cloudflare strengths confirmed in the March 10, 2026 changelog and docs:

- async crawl jobs
- HTML, Markdown, and JSON output
- depth and limit control
- include and exclude patterns
- optional JS rendering
- incremental crawl controls
- robots.txt and crawl-delay compliance

Cloudflare limits:

- no CAPTCHA or WAF bypass
- no replacement for licensed data
- best on official or documentation-style sites, not heavily defended commercial properties

## Recommended crawl pattern

1. Check `robots.txt`.
2. Start with a low-limit crawl profile.
3. Save raw responses to `data/landing/crawl/`.
4. Normalize documents into `data/raw/athletics_sites/` or `data/raw/news_sentiment/`.
5. Promote stable structured entities into `data/staged/`.
6. Generate leak-safe feature views in `data/features/`.

## Profiles included

- `configs/crawl_profiles/official-athletics-markdown.json`
  - default for roster, staff, and press-release harvesting on official sites
- `configs/crawl_profiles/news-archive-dynamic.json`
  - for JS-heavy archives where the content requires rendering

## Local helpers

- `scripts/run_cloudflare_crawl.sh`
  - starts or polls a Cloudflare crawl job using one of the included profiles
- `scripts/verify_project_setup.py`
  - checks the required directory tree and local command availability

## Optional future additions

- Firecrawl MCP if you later want a managed crawl/search/extract layer and are comfortable adding a SaaS dependency.
- `@cloudflare/playwright-mcp` if you later want a remote MCP endpoint backed by Cloudflare Browser Rendering and are willing to deploy a Worker-backed service.

## Initial next implementation steps

1. Build direct collectors for NCAA hidden feeds and `stats.ncaa.org`.
2. Add a roster and staff harvester for official athletics domains.
3. Add an odds collector.
4. Build PDF extraction for media guides and game notes.
5. Add text sentiment processing for pressers, beat coverage, and official notes.

