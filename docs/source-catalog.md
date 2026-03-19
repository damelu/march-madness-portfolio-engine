# Source Catalog

This catalog separates sources into three buckets:

1. Primary structured sources.
2. Predictive-quality rating and market layers.
3. Soft-signal and crawl-heavy sources.

## Tiering

- `Tier 1`: backbone sources for schedules, games, rosters, and official history.
- `Tier 2`: high-value enrichments for predictive power.
- `Tier 3`: noisy or brittle sources used for edge cases and proxy features.

## Recommended core stack

- NCAA hidden feeds via `data.ncaa.com`.
- `stats.ncaa.org`.
- `hoopR` / SportsDataverse for historical play-by-play and box score support.
- The Odds API for market priors.
- Official athletics sites for rosters, staff, bios, media guides, and game notes.
- Bart Torvik, KenPom, EvanMiya, Haslametrics, and Warren Nolan for predictive and committee-facing layers.

## Detailed source notes

### Official and structured

| Source | Role | Access | Notes |
|---|---|---|---|
| `data.ncaa.com` hidden feeds | Official-ish machine-readable scores, schedules, and game state | Direct HTTP | Best free backbone for game state. Undocumented, so cache aggressively and expect endpoint drift. |
| `stats.ncaa.org` | Deep historical team, player, and season stats | Web | Strong historical coverage. Treat as rate-limited. |
| `hoopR` / SportsDataverse | Historical play-by-play and ESPN-linked metadata | Package/data download | Best open historical ingest layer. Validate against source when possible. |
| SportsDataIO | Paid current structured backbone | API | Best all-in-one paid upgrade for teams, players, injuries, odds, schedules, and projections. |
| Sportradar NCAAMB | Enterprise current and historical backbone | API | Clean schemas and strong operational quality, but expensive. |
| The Odds API | Current and historical market priors | API | Practical choice for spreads, moneylines, totals, and some historical coverage. |

### Predictive and ranking layers

| Source | Role | Access | Notes |
|---|---|---|---|
| KenPom | Gold-standard adjusted team strength | Subscription web | High-value predictive layer. Respect terms and do not treat as a bulk crawler target. |
| Bart Torvik | Team, player, trend, and coaching-change layers | Web | Excellent free signal. Some high-value endpoints are disallowed by `robots.txt`. |
| EvanMiya | Player-centric value and continuity | Mixed | Strong player-level signal layer. Some data is gated. |
| Haslametrics | Complementary predictive ratings | Web | Useful consensus layer. |
| Warren Nolan | NET history, schedules, team sheets | Web | Great committee-context layer. |
| Massey Ratings | Consensus-style rating layer | Web | Good for blended priors. |
| Bracket Matrix | Selection consensus before the bracket | Web | Best public bracketology consensus source. |

### Roster movement, injuries, and morale proxies

| Source | Role | Access | Notes |
|---|---|---|---|
| Official athletics sites | Canonical rosters, staff directories, bios, media guides, and press releases | Web/PDF | Best source for current personnel and staff continuity. |
| Official game notes PDFs | Injury hints, probable starters, streaks, travel, quotes | PDF | Highest-value soft-signal source that remains reproducible. |
| RotoWire | Structured injury surface | Mixed | Useful, but not official. Prefer a licensed injury feed when possible. |
| Verbal Commits | Transfer and commitment history | Web | High-value public transfer tracker. Must be cross-checked. |
| On3 / 247Sports | Recruiting, portal, and NIL proxies | Mixed | Best public recruiting and portal context, but some coverage is premium or opinion-driven. |
| Local beat writers and pressers | Morale, discipline, chemistry, coach-status hints | Web/RSS | Useful only with source scoring and text confidence. |

## Crawl feasibility and guardrails

Robots checks performed on March 17, 2026 from this environment:

- `barttorvik.com/robots.txt`: several data-rich endpoints are disallowed, and `Crawl-Delay: 10` is present. Use allowed pages or package-based alternatives, not broad crawling.
- `sports-reference.com/robots.txt`: several college basketball paths are disallowed. Use sparingly and avoid building a crawler around disallowed endpoints.
- `on3.com/robots.txt`: general crawl is allowed outside blocked auth and board paths.
- `rotowire.com/robots.txt`: restrictive bot controls are present; treat as a manual or licensed source, not a primary crawler target.
- `verbalcommits.com/robots.txt`: returned a `403` from this environment. Assume targeted or manual collection only until verified otherwise.
- `ncaa.com/robots.txt`: broad crawl is possible on allowed sections, with many standard CMS exclusions.

## Collection policy

- Prefer APIs and package exports over crawling whenever the same entity is available in structured form.
- Use Cloudflare Browser Rendering `/crawl` only on domains that permit crawling and benefit from site-wide discovery.
- Use Playwright MCP for dynamic pages, selector debugging, and one-off extractions where static crawling fails.
- Track source reliability per field:
  - `official`
  - `licensed_structured`
  - `community_structured`
  - `public_proxy`
  - `sentiment_proxy`

## High-value gaps

- Team-level and player-level injuries remain fragmented.
- Assistant and performance staff quality mostly requires custom scraping and manual validation.
- Locker-room morale is not directly observable and must be represented with weak, explicitly noisy proxies.

