# Data Requirements

This document defines what should be collected before modeling. The objective is not just game prediction, but bracket prediction under two separate timing cuts:

- `selection_sunday`: only information available when the field and seeds are set.
- `pre_tipoff`: adds late-breaking injuries, market moves, travel, and confirmed availability.

## Canonical grains

- `team_season`
- `team_game`
- `player_season`
- `player_game`
- `coach_season`
- `matchup_game`
- `selection_sunday_snapshot`
- `pre_tipoff_availability_snapshot`

## Feature families

### 1. Team strength and results

- Record quality: overall, conference, home, away, neutral, Quadrant records, record versus top-25, top-50, and top-100 teams.
- Margin profile: average and median scoring margin, volatility, close-game record, blowout rate.
- Adjusted efficiency: adjusted offense, defense, net efficiency, tempo, strength of schedule.
- Rolling form: last 5 and 10 games, post-January splits, conference tournament form.
- Consistency: possession-level variance, turnover volatility, 3-point volatility, foul volatility.
- Neutral-court performance: neutral-site record, neutral-site efficiency, multi-team event results.

### 2. Team style and the four factors

- Shooting: eFG%, 2P%, 3P%, FT%, 3PA rate, shot-zone mix, rim rate, midrange rate, shot-quality proxies.
- Ball security: turnover rate, live-ball turnovers, press-break efficiency, forced turnover rate.
- Rebounding: offensive rebound rate, defensive rebound rate, second-chance points for and against.
- Free throws and fouls: foul drawn rate, foul committed rate, FT rate, bench foul depth.
- Transition versus half-court: frequency and efficiency for both, after-timeout efficiency, late-clock efficiency.
- Scheme proxies: zone usage, press frequency, switching behavior, pick-and-roll coverage proxies when available.

### 3. Player talent, usage, and depth

- Core production: minutes share, usage, offensive and defensive ratings, BPM-style metrics, win shares, on/off impact.
- Scoring profile: shot mix, rim finishing, pull-up shooting, catch-and-shoot efficiency, free-throw drawing.
- Playmaking: assist rate, AST/TO, creation burden, pick-and-roll ball-handler efficiency.
- Defense: steal rate, block rate, defensive rebound rate, foul rate, matchup versatility.
- Experience: class year, career starts, NCAA tournament games, continuity of minutes from prior seasons.
- Lineup context: closing lineup membership, backup point guard availability, bench net rating, star concentration.

### 4. Availability, injuries, and roster volatility

- Current status: out, doubtful, questionable, probable, minute restriction risk.
- Recent missed games and practice absence proxies.
- Hidden injury proxies: sudden minute dips, usage dips, mobility-related efficiency drops.
- Rotation disruption: replacement quality, lineup churn, bench promotion cascade.
- Discipline and eligibility: suspensions, dismissals, eligibility rulings, redshirt changes.
- Fatigue: minute overload, overtime accumulation, short-bench exposure.

### 5. Coaching and staff

- Head coach quality: career wins, tournament wins, seed-relative tournament overperformance.
- Preparation: performance with extra rest, short rest, rematches, after losses.
- In-game management: timeout patterns, foul and substitution patterns, close-game behavior.
- Staff continuity: years together, associate head coach continuity, recent staff turnover.
- Player development: year-over-year improvement under the staff, transfer improvement.
- Scheme adaptability: performance against tempo, zone, press, and switching environments.
- Performance staff proxies: continuity of sports medicine and strength staff where public.

### 6. Morale, psychology, and cohesion proxies

These are weak signals and should always carry a confidence score.

- Team continuity: returning minutes, returning possession share, lineup continuity.
- Quote sentiment: coach and player press conference sentiment, public confidence or tension.
- Beat-writer tone: article sentiment, conflict reports, hot-seat or distraction narratives.
- Leadership proxies: veteran guard minutes, captains, upperclass starter count.
- Adversity response: performance after losses, after major injuries, and in hostile environments.
- Emotional swing markers: bubble drama, auto-bid celebrations, conference tournament letdown spots.

### 7. Recruiting, transfers, and development pipeline

- Recruiting pedigree: player recruiting ranks, class strength, roster talent index.
- Transfer pedigree: transfer rankings, prior-school quality, prior production.
- Development curve: year-over-year usage and efficiency growth.
- Age profile: roster age weighted by minutes, older transfer count, international veteran presence.
- NBA-caliber talent: projected draft picks, all-conference honors, high-end shot creators, rim protectors.

### 8. Matchup-specific features

- Style clash: tempo differential, shot-profile mismatch, ORB versus DRB weakness, FT-rate mismatch.
- Positional matchup: size and length by position, ball-handler advantage, rim protector versus rim dependence.
- Shooting geometry: preferred shot zones versus opponent-denied zones.
- Ball-screen matchup: pick-and-roll efficiency versus opponent scheme vulnerability.
- Zone and press exposure: offense against zone or press, opponent tendency to use them.
- Similar-opponent comps: historical performance against stylistically similar teams.

### 9. Schedule, rest, travel, and environment

- Days since last game, conference tournament load, overtime burden.
- Miles traveled, time zones crossed, altitude, pseudo-home fan advantage.
- Venue familiarity: arena type, dome or unusual sightline experience.
- Season wear: games played, recent travel density, average starter load over the prior month.

### 10. Officiating and whistle environment

- Team foul sensitivity and foul-drawing dependence.
- Referee crew tendencies when assignments are available.
- Interaction effects between physical teams and tight-whistle environments.

### 11. Rankings, committee priors, and markets

- Committee-facing metrics: NET, KPI, SOR, WAB, BPI, seed projections.
- External ratings: KenPom, Bart Torvik, Haslametrics, Massey, Warren Nolan.
- Betting markets: open and close spread, moneyline, total, futures, line movement.
- Public pick rates after the bracket is released for bracket-EV models.

### 12. Historical tournament and program context

- Program seed history, tournament appearance history, seed-relative overperformance.
- Coach March history and upset record.
- Conference tournament and NCAA tournament over- or under-performance.
- Path difficulty by region after the bracket is known.

## Recommended priority order

1. Adjusted team strength and schedule strength.
2. Market priors and close-line information.
3. Player availability and rotation health.
4. Guard creation, turnover control, and late-game offense.
5. Matchup-style interactions.
6. Coaching quality and staff continuity.
7. Travel, rest, and geography.
8. Morale and sentiment proxies with explicit confidence weighting.

## Collection rules

- Always store raw captures before parsing.
- Timestamp every row with source and freshness metadata.
- Mark every field by availability cutoff: `selection_sunday`, `pre_tipoff`, or `postgame_only`.
- Never leak post-selection outcomes into selection-time features.
- Treat morale, chemistry, and hidden-injury inputs as probabilistic features, not ground truth.

