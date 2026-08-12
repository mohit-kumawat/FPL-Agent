# fpl-agent

A Fantasy Premier League decision system built to be **operated by a headless AI
agent**. A deterministic Python pipeline does the maths — expected points, squad
optimisation, decision thresholds — and an agent does the judgement: live team
news research, memory, and talking to you. Every recommendation is source-tagged,
logged, and reproducible.

Validated by point-in-time replay of past seasons, with no hindsight, enforced
structurally rather than by convention. Replaying 2024/25 through the exact
pipeline scores **2,288 points** under real matchday rules — automatic subs and
vice-captain fallback, no chips (2,366 on 2023/24) — against an overall average
of roughly 2,100. An ablation ladder attributes the edge: the full model beats
a naive PPG strategy by +187/+238 points with bootstrap CIs excluding zero on
both seasons — see [§10](#10-validation--what-is-actually-proven) for what that
does and does not prove.

```bash
git clone https://github.com/<you>/fpl-agent && cd fpl-agent
uv sync
cp squad.example.yaml squad.yaml     # your team (gitignored); leave players: [] to start
uv run fpl status                    # where we are in the season, what's pending
uv run fpl build                     # optimal 15 under £100m / 2-5-5-3 / max-3-per-club
uv run fpl daily                     # the daily loop -> reports/YYYY-MM-DD.md + .json
```

Then point an agent (Claude Code, or your own) at **[AGENT.md](AGENT.md)** — the
operating contract covering what to run, when to research, and what it may never do.

**What it does not do:** it never logs into your FPL account, never submits
transfers or plays chips, and never handles passwords. It reads public endpoints
and produces recommendations for a human to act on.

| | |
|---|---|
| Docs | This file (full reference) · [AGENT.md](AGENT.md) (agent runbook) · [CONTRIBUTING.md](CONTRIBUTING.md) |
| Evidence | [eval/2024-25-report.md](eval/2024-25-report.md) · [eval/strategy-sim-report.md](eval/strategy-sim-report.md) · [eval/ablation-report.md](eval/ablation-report.md) |
| Tests | `uv run pytest -q` (84 offline tests) · `uv run python eval/run_backtests.py` (replay suite) |
| Licence | MIT |

*The rest of this file is the complete technical reference: exactly what the code
computes today, and what the operating agent must do. Where this document and the
code disagree, the code wins — then fix this document. Known gaps are stated
plainly in [§11](#11-known-limitations); nothing is claimed that the eval suite
does not measure.*

---

## 1. What this is

- **Deterministic pipeline** (`fpl_agent/`, ~3,400 lines, Python 3.13 + pandas +
  scikit-learn + PuLP): data collection, change detection, expected points, squad
  optimisation, Monte Carlo outcomes, decision policy, reports, memory. Same
  input, same output, always (simulation included — it is seeded).
- **Headless agent** (any capable LLM session following `AGENT.md`): runs the
  pipeline, does live web research, writes findings as *signals* the models
  consume, compares today's recommendation against past decisions, messages you.

Every claim in the output carries its epistemic source: `[DATA]` (a fact from the
API), `[MODEL]` (a computed prediction), `[SIGNAL]` (research judgement),
`[RECOMMENDATION]` (the final call), `[MEMORY]` (comparison to a past decision).

Module map:

| Module | Lines | Responsibility |
|---|---|---|
| `data.py` | 365 | API client, snapshots, price-freshness logic, change detection, history loaders |
| `models.py` | 329 | Minutes model + distribution (p_start/p_60/p_bench), prior tier, ensemble, component EP |
| `policy.py` | 347 | Decision thresholds, captaincy + strategy modes, chip EV, price timing, flags |
| `memoryio.py` | 318 | State, decisions, predictions, signal roles/validation, chip scenarios |
| `replay.py` | 253 | Point-in-time historical reconstruction with leakage assertions |
| `report.py` | 239 | Markdown + JSON report writer |
| `daily.py` | 231 | Orchestrator: verify → detect changes → trigger matrix → policy |
| `optimizer.py` | 193 | Three PuLP integer programs (build / XI / transfers) |
| `simulate.py` | 178 | Seeded Monte Carlo point distributions (haul/blank tails, captaincy) |
| `features.py` | 160 | Fixture channels, set-piece flags, rolling form |
| `lifecycle.py` | 160 | Gameweek stage machine and pending items |
| `verify.py` | 140 | Pre-flight state verification (blocking) |
| `scoring.py` | 91 | Real matchday rules: autosubs + vice-captain fallback |
| `cli.py`, `rating.py`, `backtest.py`, `config.py` | 389 | Entry points, squad grading, model backtest, constants |

---

## 2. Data layer

| Source | What | When fetched |
|---|---|---|
| `bootstrap-static` | 577 players: prices, status, `chance_of_playing`, news, season aggregates (minutes, starts, xG, xA, xGI, xGC, ICT, BPS, `points_per_game`, `defensive_contribution`), set-piece orders, ownership, transfer counts; 20 teams with strength ratings; 38 events; chip windows | Every run, subject to the freshness rules below |
| `fixtures/` | All fixtures with per-side FDR | Alongside bootstrap |
| `element-summary/{id}/` | Per-player gameweek history, current season | Only once a new gameweek has finished (~577 calls, cached to parquet) |
| `entry/{id}/event/{gw}/picks/` | Your actual picks | From GW1 onward; public, needs `entry_id` |
| vaastav CSVs | Historical player-gameweek panels | Once, cached as parquet. **Verified available for 2023/24, 2024/25 and 2025/26** (2025/26 loads 29,757 rows across 38 gameweeks and includes `defensive_contribution`). 2026/27 does not exist there, which is why the current season is built from `element-summary`. |
| `squad.yaml` | Your squad with **purchase prices**, bank, free transfers, chips | Every run |
| `signals/*.yaml` | Agent research findings | Every run, subject to validation and TTL |

**Change detection** diffs each day's bootstrap against the previous snapshot into
price changes, status changes, news changes and gameweek-state transitions. Those
diffs drive the trigger matrix ([§6](#6-the-daily-loop)).

### 2.1 Price freshness

Prices come only from the official FPL API, never a scraped mirror. For 2026/27,
FPL applies price changes at **00:00 UK local time**
([source](https://www.premierleague.com/en/news/4680462/whats-new-in-202627-fantasy-price-change-predictor)).
That is a *local wall-clock* time, so the UTC instant moves: 23:00 UTC under BST,
00:00 UTC under GMT. The boundary is therefore computed in `Europe/London` and
converted, with a 20-minute grace period for the API to settle. A fixed UTC
constant served pre-change prices for hours during BST while reporting them fresh;
that is now a regression test.

A refetch happens when any of these holds (`data.is_stale`):

| Condition | Result |
|---|---|
| No snapshot today | fetch |
| Snapshot predates the last 00:00-UK boundary | **refetch** — the price-change guarantee |
| Snapshot older than `MAX_SNAPSHOT_AGE_HOURS` (12h) | refetch |
| Within 24h of a deadline and older than `DEADLINE_FRESH_HOURS` (3h) | refetch — the bar tightens to match what verification demands |
| `fpl refresh` or `fpl daily --force` | refetch unconditionally |

Each snapshot carries a `.meta.json` sidecar recording the true fetch timestamp
and source URL; every report states when prices were fetched and whether that run
refetched. Within 24h of a deadline, data older than three hours **blocks
recommendations** rather than advising on stale prices.

FPL's Price Change Predictor (`price_change_percent` in bootstrap) feeds timing
advice on recommended transfers — act-tonight on imminent rises, wait on
imminent falls, owned-value-at-risk warnings ([§7](#7-decision-policy)). The
field's sign convention is assumed (it was all-zero preseason), and it is never
allowed to pick a transfer, only to time one.

### 2.2 Selling prices

The public API never exposes selling prices, which is why purchase prices live in
`squad.yaml`:

- price **rose**: `sell = purchase + floor((now − purchase) / 2)` — profits halved
- price **fell**: `sell = now` — losses are taken in full, with no floor at the
  purchase price

Getting the second case wrong inflates the transfer budget and makes the optimiser
propose squads you cannot afford. It did, once; it is now tested.

---

## 3. Expected points

Every player gets `ep_next`, `ep_horizon` (discounted over five gameweeks), and an
uncertainty band (`ep_sd`, `ep_p10`, `ep_p90`). Final EP is a **55/45 blend of two
paths** (`W_COMPONENT = 0.55`):

```
COMPONENT path (process — what the player does per 90):
  attack90  = xG/90 × goal_pts(pos) + xA/90 × 3     goal_pts: GK 10, DEF 6, MID 5, FWD 4
  cs90      = P(clean sheet) × cs_pts(pos)          P(CS) = exp(−xGC/90)   [Poisson]
  neutral90 = bonus/90 + saves/90 ÷ 3 + defensive-contribution − xGC/90 ÷ 2 (GK/DEF)
  app_pts   = P(play) + P(60')
              P(play) = clip(xmins / 0.60);  P(60') = logistic((xmins − 0.63) / 0.09)

  ep_next(comp) = app_pts × n_fixtures(next)
                + xmins × [attack90 × ATT_mult + cs90 × DEF_mult + neutral90 × n_fix]

REALIZED path (outcomes — what the player actually scored):
  ep_next(real) = ep_ppg × xmins × blended_fixture_mult

ep_next = 0.55 × component + 0.45 × realized + set-piece role bonus
```

Two paths because components stop overperformance being extrapolated (a 7.0-PPG
player at 0.30 xGI/90 is not a 7.0-PPG player at 0.55), while the realized path
keeps information components cannot see — bonus magnetism, penalty share already
embedded in historical xG. Measured on the 2024/25 replay, adding the component
path lifted captain actual points from 9.42 to 10.00 per gameweek.

### 3.1 `ep_ppg` — the realized path's quality estimate

A season-progress blend between a prior tier and an in-season tier:

```
w = min(1, gws_played / 8)
ep_ppg = (1 − w) × PRIOR + w × SEASON
```

**PRIOR tier** — full weight preseason, fading to zero by GW8:

1. Players with real history (minutes ≥ a threshold that self-scales early season
   as `min(900, 0.5 × max observed minutes)`):
   `PRIOR = 0.5 × points_per_game + 0.5 × ridge(xGI/90, xGC/90, ICT/90, start_rate)`.
   Exact ridge spec: target is realized PPG; the four per-90 features are
   standardised with `StandardScaler`; `sklearn.Ridge(alpha=1.0)`; one model per
   position, fit only on that position's players with real history and PPG > 0,
   skipped entirely below 10 samples (falling back to raw PPG). The shrinkage
   target is therefore *what a player with these process stats typically scores in
   this position* — a positional process prior, not the position mean.
2. No history (new signings, promoted players): the median PRIOR of the same
   position × £0.5m price bucket, falling back to position median scaled by
   relative price. These players are the model's weakest point until roughly GW3
   and are force-flagged in the report ([§4b](#4b-decision-policy)).

**SEASON tier** — an ensemble of two models, once gameweeks exist:

- **Model A (recency):** weighted mean of the last five gameweek scores, weights
  1..5.
- **Model B (process ridge):** ridge on five-gameweek rolling means of xGI, xGC,
  ICT and minutes, all shifted one gameweek so there is no target leakage, fit
  across the league's player-gameweek history **within a single season**.
- **Blend: `SEASON = 0.4 × A + 0.6 × B`.** λ was grid-searched across *two*
  seasons' out-of-sample windows and the ranking flips between them — 2024/25
  favours recency, 2025/26 favours ridge — so no extreme is trustworthy. 0.6 is
  the maximin choice. An earlier single-season λ = 0.7 is recorded in
  `learnings.md` as a documented overfitting mistake.

Seasons are never pooled into one training set: Model B fits within a season, and
each backtest season runs independently. The remaining cross-regime exposure is
narrow but real — prior-season PPG feeding the cold-start prior across FPL rule
changes (defensive contributions arrived in 2025/26, assist rules changed, BPS
changed again for 2026/27). It decays to zero by GW8. See [§11](#11-known-limitations).

### 3.2 `xmins` — expected minutes

Points require pitch time, so this gates everything.

```
in-season:  xmins = clip(mean(last 5 GW minutes) / 90, 0, 1) × chance_of_playing
preseason:  xmins = (0.55 × starts/38 + 0.45 × minutes/(38×90)) × chance_of_playing
no history: xmins = clip((price − 4.0) / 10, 0.30, 0.85)
```

Note **mean**, not total: a player averaging 80 minutes gets 0.89, one averaging
25 minutes gets 0.28. The 0.55/0.45 start-rate blend keeps nailed-when-fit players
who missed a stretch through injury from being priced as rotation risks.
`chance_of_playing` comes from the API's injury flags. Signals can override
`xmins` directly, which is how "not in the predicted XI" enters the model.

### 3.3 Fixtures — per channel, not one number

A fixture that is hard for a forward is not equally hard for a goalkeeper. From
FPL's team strength ratings: `ATT_mult = my venue attack ÷ opponent venue
defence`, `DEF_mult = my defence ÷ opponent attack`, each clipped to [0.70, 1.40].
Horizon sums are discounted 0.90 per gameweek so near-term fixtures dominate.
Doubles sum both matches; blanks contribute zero — this is how the GW24 double and
a postponed fixture were both handled correctly in backtests. The FDR table
survives only as a fallback and for risk reporting.

### 3.4 Set pieces and uncertainty

**Set-piece bonus is deliberately small** (+0.2 first-choice penalty taker, +0.05
others): an incumbent taker's penalty output is already inside his historical xG,
so a large injection double-counts. A player who *newly gains* the duty should get
an explicit signal instead.

**Uncertainty:** `ep_sd = 1.3 × [1.9 × √(attack90 × xmins) + 0.55 × √cs90 + 0.3]`
— a parametric approximation where attacking returns are Poisson-like, clean
sheets binary, appearances near-deterministic. The 1.3 factor is empirically
calibrated: uncalibrated bands covered 68% of top-50 actuals against an 80%
target; after scaling, coverage is 0.83, re-measured on every eval run. Bands are
`ep_next ∓ 1.28 × ep_sd`, a normal approximation adequate for captain-risk framing
but not for tail bets.

**Calibration:** every run stores predictions; when a gameweek finishes they are
scored (MAE overall and top-50). Knob changes stay human-in-the-loop through the
VALIDATED tier of `learnings.md` — there is no silent online updating, because
four to eight in-season data points is how you fit noise.

---

## 4. Optimisation

Three PuLP integer programs, all enforcing the full ruleset: £100m, squad of
2 GKP / 5 DEF / 5 MID / 3 FWD, XI with 1 GKP and 3–5 DEF and 2–5 MID and 1–3 FWD,
maximum three per club, captain must start.

1. **`build_squad`** (fresh 15, for GW1 or a wildcard): maximises
   `Σ ep×XI + ep×captain + 0.1 × Σ ep×bench`. Bench is weighted at 0.1 so money
   concentrates in the XI; captaincy sits inside the objective so premiums are
   valued properly. Supports `--lock` to force players in, and bans.
2. **`pick_xi`** (fixed 15 → lineup): maximises XI + captain EP, returning
   formation, captain, vice, and bench order (goalkeeper first, then by EP).
3. **`plan_transfers`**: solves once per transfer count k with "exactly k leave",
   budget = bank + Σ selling prices, buying at market price and keeping at selling
   price. Compares `objective − 4 × max(0, k − free_transfers)` across k.
   **Search depth defaults to `free_transfers + 1`, capped at 6** — free transfers
   bank up to five, and "4 transfers is not optimal" must not be confused with
   "4 transfers was never evaluated".

### 4b. Decision policy

The optimiser proposes; policy disposes. An unconstrained optimiser will happily
churn three transfers for +0.4 EP.

| Decision | Rule |
|---|---|
| Free transfer | ≥ **+2.0 EP** over five gameweeks per move. **Dynamic:** ×1.5 before GW4 (noisy data), ×0.75 when 4–5 transfers are banked (banking further wastes them) |
| Points hit (−4) | ≥ **+6.0 EP** net of the hit, same dynamic scaling |
| Hit fitness rule | a hit may never buy a player with `xmins < 0.6` unless the gain clears the bar by a further 25%, and then it is flagged for manual fitness check |
| Plan choice | among plans clearing their threshold, the **highest net gain** wins, tie-breaking toward fewer transfers |
| Otherwise | **hold** and bank the transfer; the report always states the net EP of the best alternative |
| Captain | highest `ep_next` in the XI, with ownership context; a differential option (ownership < 15%, within 1 EP) is shown advisory-only |
| Chips | windows flagged (wildcards GW2–19 and GW20–38, doubles for TC/BB); recommend-only, the owner decides |
| Uncertainty flags | before GW3, any pick at £7.0m or above without a season of data **and** without a signal is flagged "research before trusting" |

### 4c. Squad rating

`fpl rate` grades your 15 against the unconstrained optimal build. Overall is your
best XI's EP as a percentage of the optimal XI's (A+ ≥ 97%, A ≥ 92%, B+ ≥ 85%,
B ≥ 78%, C+ ≥ 70%, C ≥ 60%, else D). Per player: EP against the best same-position
player within ±£0.5m. Plus risk flags for three-per-club concentration, injury
flags, an expensive bench, and a hard fixture run.

---

## 5. Signals — how research enters the model

Files in `signals/`, written by the agent after web research, re-read every run:

```yaml
date: 2026-08-15
source: "Arteta press conference"
confidence: high        # weights ep_per_gw by 1.0 / 0.6 / 0.3
ttl_days: 14            # or an explicit `expires: 2026-08-22`
notes: "Saka in full training, expected to start GW1"
adjustments:
  - player_id: 12       # FPL element id
    xmins_min: 0.9      # floor on the minutes fraction — "nailed"
    reason: "confirmed starter, was priced as a rotation risk"
  - player_id: 388      # a different player, NOT expected to start
    xmins_max: 0.45     # cap — "not in the predicted XI"
    reason: "omitted from the predicted XI"
```

Use `xmins_min` **or** `xmins_max` per player, never both — a floor above a cap
rejects the whole file.

Discipline: signals carry **minutes facts**, not opinions — the agent tells the
model who is *playing*, never who is *good* (the blind-label test in
`eval/agent-backtest-report.md` measured agent quality-opinions at negative
value). `ep_per_gw` remains only for the rare quality/role fact minutes can't
express (pen duty gained or lost, position change): it needs a direct quote in
`source` and should stay within ±0.5 even though validation allows ±2. Minutes
bounds are not confidence-weighted, since they are facts or they should not be
written.

**Validation is enforced, not advisory.** A file is ignored entirely, and the
reason printed in the report, if it has expired, fails to parse, sets a floor
above a cap, puts a bound outside [0, 1], or exceeds the ±2 EP nudge limit.
Contradictions *across* files drop both bounds. An autonomous operator must never
depend on a human remembering to delete a stale file.

---

## 6. The daily loop

`fpl daily` verifies state, diffs against yesterday, then does only the work the
day's changes justify:

| Change detected | Work done |
|---|---|
| Nothing, deadline more than three days out | **One-line log. No models.** |
| Price changes only | Squad value and selling prices updated; no model rerun |
| Status or news on an owned or shortlisted player | Minutes model and optimiser only |
| A gameweek finished | Full retrain: fetch element-summaries, rebuild the panel, **score stored predictions**, recompute |
| A new file in `signals/` | Merge signals, then optimise |
| Deadline within 72h | Full decision run: transfers, captain, bench, chips |
| Deadline within 24h | Final-check mode with act-now flags |

Same-day reruns no-op unless forced. A **verification-blocked run is logged under
its own kind**, so it does not mark the day complete and the fix-and-retry the
blocker asks for actually runs.

### 6b. Lifecycle stages

`PRESEASON → PLANNING → DEADLINE_SOON (≤72h) → DEADLINE_IMMINENT (≤24h) →
GW_LIVE → POST_GW → PLANNING …`, each carrying a playbook and auto-seeding
`memory/pending.json`. Filling in `squad.yaml` auto-completes the preseason items.

---

## 7. Memory

| File | Purpose |
|---|---|
| `memory/learnings.md` | Three-tier evidence log — **tracked in git** |
| `memory/state.json` | Stage, gameweek state, signals already seen |
| `memory/pending.json` | Open action items |
| `memory/decisions.jsonl` | Every recommendation; each report diffs against the last |
| `memory/runs.jsonl` | What ran when, and which triggers fired |
| `memory/predictions/gwXX.csv` | Stored forecasts, auto-scored after the gameweek |
| `memory/research/*.md` | Dated research notes with sources and trust calls |

`learnings.md` separates **VALIDATED** (tested via the eval suite; only these may
justify a config change), **OBSERVED FACTS**, and **HYPOTHESES** (never act on
these). Without that separation an LLM writing "lessons" turns a hunch into
folklore and folklore into a model constant.

---

## 8. Reports

`reports/YYYY-MM-DD.md` plus a machine-readable `.json` twin:

1. **Stage** — lifecycle state, deadline countdown, playbook, pending items
2. **Verification** — price provenance and freshness, state checks, any blockers
3. **What changed** — price, status and news diffs
4. **Models ran** — or explicitly "None", with the reason
5. **Key findings** — squad rating, calibration scores, risks, signals (including
   any that were *ignored* and why)
6. **Recommended action** — buy/sell/hold with reasons and net EP, XI, formation,
   bench order, captain and vice with confidence, chip notes, and a `[MEMORY]`
   comparison against the previous decision
7. **Uncertainty flags**, **Long-term outlook**, **Monitor next**

---

## 9. Operating the agent

**Verify first, analyse second, recommend last.** Full contract in `AGENT.md`.

**Phase 0 — orient.** `uv run fpl status`.

**Phase 1 — verify (automatic, blocking).** API sanity; `squad.yaml` legality
(exactly 15, no duplicates, ids that exist, 2-5-5-3, ≤3 per club, purchase prices
present, free transfers 0–5, bank ≥ 0, valid chip names); **reconciliation against
the official picks endpoint** once `entry_id` exists, where a player mismatch or a
bank mismatch blocks; price freshness. A blocked run still writes a report — with
blockers and no recommendation.

**Phase 2 — detect changes** from official FPL data.

**Phase 3 — run only what the trigger matrix requires.**

**Phase 4 — judgement, the agent's real job.** Read Key findings and Monitor next;
research flagged injuries, press conferences and predicted lineups; cross-check
sources and record trust calls in `memory/research/`; write findings to
`signals/*.yaml`; rerun once with `--force` if signals changed.

**Phase 5 — recommend and remember.** Compare against `decisions.jsonl` and
explain any flip; message the owner with buy/sell/hold, reasons, net EP, captain,
confidence and what is being watched; update memory.

### CLI

```
uv run fpl status                 # stage, playbook, pending, last decision (JSON)
uv run fpl daily [--force]        # the daily loop
uv run fpl build [--lock 411,426] [--budget 100]
uv run fpl rate                   # grade squad.yaml against optimal
uv run fpl pending [list|add <text>|done <substr>]
uv run fpl backtest               # model A/B/ensemble, out-of-sample
uv run fpl refresh                # force re-fetch
uv run pytest -q                  # 84 offline tests, no network
uv run python eval/run_backtests.py    # point-in-time replay suite
uv run python eval/strategy_sim.py     # full-season strategy return
```

---

## 10. Validation — what is actually proven

`replay.py` reconstructs exactly what was knowable before each historical
deadline: panel rows strictly `round < GW`, prices last-observed before the
gameweek, prior-season aggregates for cold start. Hindsight is structurally
impossible and asserted in the suite.

**Prediction quality** (2024/25, GW6–24, starters pool `xmins > 0.3`):

| Metric | Value |
|---|---|
| Captain actual | 10.21 pts/GW, 63% of the achievable top-6 ceiling |
| Top-11 actual | 5.59 pts/player/GW |
| Spearman ρ (ep_next vs actual) | 0.364 — GKP 0.38, MID 0.35, DEF 0.34, FWD 0.24 |
| ρ by price | ≤£5.5m 0.34, £5.5–7.5m 0.28, >£7.5m 0.24 |
| p10–p90 coverage | 0.83 against an 0.80 target |

**Strategy return** (GW1 build → weekly policy-gated transfers → XI + captain):
under real matchday rules (automatic substitutions + vice-captain fallback)
**2,288 points on 2024/25** (+387 against holding the GW1 squad) and **2,366 on
2023/24** (+394). The legacy no-autosub scores — 2,232 and 2,267 — are still
reported for comparability with earlier claims. Both above the overall average
of roughly 2,030–2,100, both below top-10k pace of roughly 2,450.

**Ablation ladder** ([eval/ablation-report.md](eval/ablation-report.md)):
`ppg → +fixtures → +minutes → full → full-greedy`, all arms through the same
season loop and scorer, evidence as paired per-GW differences with a seeded
bootstrap 90% CI. The whole ladder — `ppg → full` — earns +187 (2024/25, CI
[+1.0, +8.7] per GW) and +238 (2023/24, CI [+2.1, +10.3]): the model's edge
over naive PPG is real on both seasons. Individual rungs are mostly within
noise on single seasons (only `+minutes → full` clears its CI, on 2023/24),
and threshold-free greedy transfers flip sign by season (+44/−46) — the
maximin case for keeping the policy gate.

**Named decisions the replay got right:** switched captaincy permanently to Salah
from GW11, three weeks before the official archive calls GW14 the turning point;
deprioritised Haaland before an eight-gameweek drought; recommended Triple Captain
Salah on the GW24 double with sound expected value (actual 29 → 87 chip points);
ranked Bowen the top GW38 captain on form and fixture (actual 13, beating every
premium).

**What this does not prove.** The simulation still includes no chips (real
managers gain roughly 30–60 points from them), the historical archive carries
no injury flags (the replay assumes availability), and two seasons is a small
sample — the ablation CIs quantify that rather than remove it. The absolute
total is a *strategy replay score*, not a claim that this manager would have
finished on 2,288 points.

The suite has caught five real bugs before they reached anyone: an early-season
NaN collapse, NaN poisoning in the blend, substring name matching, a selling-price
error that inflated the transfer budget, and a metadata-glob collision that would
have killed the daily loop on its second day. It is the regression suite; rerun it
after any model change.

---

## 11. Known limitations

Stated plainly, because a system that hides these is more dangerous than one that
underperforms.

- **Cold start is still the weakest point.** New signings and promoted players
  are essentially a price prior until roughly GW3. Mitigated — not solved — by
  rules-aware priors, mandatory uncertainty flags at every price point, and
  structured signals.
- **Scoring-regime drift is now downweighted, not modelled.** `RULES_VERSION`
  maps seasons to rule generations and shifts the cold-start prior from realized
  PPG toward rule-independent process stats across a boundary (0.5 → 0.35). The
  0.35 weight is a conservative pick that cannot be validated until the new
  regime has real gameweeks; the ablation suite is the tool for revisiting it.
- **Minutes modelling is still reactive.** The model now emits a
  P(start)/P(60)/P(bench) distribution with an uncertainty spread, and signals
  carry a structured role vocabulary (`expected_starter`, `managed_minutes`,
  `not_in_predicted_xi`, …) — but the inputs remain rolling minutes, injury
  flags, and research. A missed press conference is still a real failure mode;
  nothing anticipates rotation the news has not hinted at.
- **`P(CS) = exp(−xGC/90)` assumes Poisson goals**, and the Monte Carlo engine
  *explicitly inherits* that assumption — simulation made it stochastic, not
  better. Doubles are treated as one aggregate exposure, which slightly
  understates per-match clean sheets. A correlated-scoreline model
  (Dixon-Coles) is the genuine fix and is not built.
- **Tail probabilities now come from Monte Carlo** (captaincy: median, p10/p90,
  haul/blank). The normal ep_sd band still backs the report's p10–p90 columns
  and remains a rough approximation there.
- **Small point sources are simulated, not predicted**: cards, own goals and
  penalty events enter the Monte Carlo tails at league-ballpark rates, but
  per-player discipline (a Casemiro yellow rate) is not modelled and the EP
  mean ignores them entirely.
- **xG and xA are mapped to FPL goals and assists directly**, though FPL's
  assist definition is broader than standard xA.
- **Chip EV covers Triple Captain and Bench Boost only**, computed *given*
  double/blank scenarios the agent supplies from research (plus a decaying
  default prior). Free Hit and Wildcard remain structural advice; nothing
  optimises a chip route across the whole season.
- **Ownership enters captaincy only.** `safe` / `balanced` / `chase` modes can
  hand the armband to a close differential, but transfers and squad structure
  are ownership-blind, and effective ownership (top-10k) has no data source.
- **The transfer horizon is five gameweeks of expected points.** It does not
  optimise team value, future purchasing power, or a route into a future
  premium. Price-change timing advice exists, but it decorates moves the model
  already wants rather than planning around the market.
- **Set-piece duty is a flat bonus**, not a modelled penalty rate.
- **FWD and premium ranking is weakest** (ρ ≈ 0.24); treat single-gameweek premium
  comparisons as near-ties.
- **The replay assumes availability**, since the historical archive carries no
  injury flags. The autosub-aware totals inherit this: a player the real manager
  knew was injured stays in the replay XI and scores zero.
- **`price_change_percent` semantics are assumed** (signed progress toward
  tonight's move) — the field was all-zero preseason, so the sign convention is
  unverified until the market opens. Thresholds are set where being wrong is
  cheap.

---

## 12. Roadmap

Ordered by expected value, kept honest by the same eval suite. The 2026-08
upgrade landed the previous roadmap's eight items (ablation ladder, autosubs,
probabilistic minutes, Monte Carlo, chip EV, strategy modes, price-predictor
ingestion, `rules_version`); what remains:

1. **A correlated-scoreline goal model** (Dixon-Coles or similar) so clean
   sheets and concessions stop being independent Poissons — the biggest
   remaining modelling approximation.
2. **Empirical threshold calibration** — the FT/hit bars and the cross-regime
   PPG weight (0.35) are conservative picks; the ablation harness can now
   measure them.
3. **Modelled set-piece rates** (penalty share × expected penalty events)
   replacing the flat bonus.
4. **Assist-definition calibration** — historical FPL-assist-over-xA residuals
   by player and team.
5. **Multi-GW transfer routes** — team value, purchasing power, and planned
   paths into premiums, beyond the five-GW EP horizon.
6. **Rank-aware transfers** — extend `chase` beyond the armband once an
   effective-ownership source exists.
7. **Forward-test accumulation** — `phase3_prereg.py score` is live; the
   2026/27 season is the first uncontaminated measurement of the whole system.

---

## 13. Configuration

All constants live in `config.py` with the evidence for their value beside them.

| Knob | Value | Basis |
|---|---|---|
| `W_COMPONENT` | 0.55 | 2024/25 replay: captain 9.42 → 10.00, capture 59% → 63% |
| `ENSEMBLE_LAMBDA` | 0.6 | Two-season maximin; single-season extremes flip |
| `HORIZON_DISCOUNT` | 0.90/GW | Near-term fixtures dominate transfer value |
| `STRENGTH_CLIP` | [0.70, 1.40] | Continuous channel multipliers |
| `COLD_START_GWS` | 8 | Prior fades linearly to zero |
| `FORM_WINDOW` | 5 GWs | Literature default |
| `HORIZON_GWS` | 5 | Transfer-decision window |
| `BENCH_WEIGHT` | 0.1 | Keeps money in the XI |
| `MAX_TRANSFER_SEARCH` | 6 | Covers five banked transfers plus a hit |
| `PRICE_CHANGE_TZ` | Europe/London | Changes land at 00:00 UK local, not a fixed UTC hour |
| FT / hit thresholds | 2.0 / 6.0 EP, dynamically scaled | Conservative; empirical calibration is a roadmap item |
| `PPG_PRIOR_WEIGHT` / `_CROSS_REGIME` | 0.5 / 0.35 | Cross-regime prior leans on process stats; 0.35 is unvalidatable before GW1, revisit via ablation |
| `PRICE_MOVE_IMMINENT` / `_WATCH` | 90 / 60 % | Timing nudges only on moves the model already wants — cheap if the field's semantics surprise |
| `CHASE_EP_TOLERANCE` | 1.0 EP | Max EP a differential armband may cost in chase mode |
| `CHIP_PLAY_MARGIN` | 1.15 | Chips are one-shot options; ties go to holding |
| `DEFAULT_DOUBLE_PROB` / `_LAST_GW` | 0.8 / GW30 | Unresearched-double prior; silence after GW30 means none |
