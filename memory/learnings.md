# Learnings

Three tiers, per the evidence-discipline rule. **Only VALIDATED entries may
change model config.** Observed facts await more data; hypotheses are folklore
until tested — the agent must not act on them.

Every entry carries a date. VALIDATED entries must also cite the evidence that
promoted them (an eval report, a metric, or a gameweek). `fpl daily` checks this
and flags violations in the report, because a file nobody prunes becomes folklore
by week 10. Tier caps: 12 / 20 / 15 — at the cap, prune or promote before adding.

## VALIDATED (tested across historical data; may drive config)

- **Component-based EP beats PPG-compression** (2026-08-11). Same 2024/25
  GW6–24 sweep, same information: captain actual 9.42 → 10.00 pts/GW, ceiling
  capture 59% → 62%, top-11 5.57 → 5.60. Also resolved the GW1 Watkins-over-Salah
  miss without outcome tuning (Saka/Salah ranked 1-2, actual 12/14). Basis of
  W_COMPONENT=0.55.
- **No ensemble extreme is robust across seasons** (2026-08-11,
  eval/strategy-sim-report.md). λ grid on 2024/25 AND 2025/26 test windows:
  rankings flip completely (A best/B worst, then B best/A worst). λ=0.6 is the
  maximin (worst-case top-11 4.14). The earlier λ=0.7 choice was overfit to one
  season — a caught mistake, kept here as a warning.
- **Logistic appearance points** (2026-08-11, eval/2024-25-report.md), P(play) +
  logistic P(60') over a hard clip: captain 10.00 → 10.21, capture 62% → 63% on
  the 2024/25 GW6-24 sweep.
- **ep_sd calibration** (2026-08-11): uncalibrated p10-p90 covered 68% of top-50
  actuals vs the 80% target → scaled sd ×1.3 → coverage 0.83. Baseline metrics for
  any future model change: rho=0.364 (starters), by position GKP .38 / MID .35 /
  DEF .34 / FWD .24.
- **Early-season data thresholds must self-scale** (2026-08-11): a fixed
  900-minute history gate produced all-NaN EP at GW9/10 — caught by the replay
  suite, fixed, now regression-tested.
- **Strategy return under real matchday rules** (2026-08-12,
  eval/strategy-sim-report.md): 2024/25 = 2,288 pts, 2023/24 = 2,366 pts with
  autosubs and vice-captain fallback; +387 / +394 vs holding the GW1 squad. The
  no-autosub figures (2,232 / 2,267) are kept only for comparison with earlier
  claims. Both above the ~2,030-2,100 average, below top-10k pace (~2,450).
- **The model's edge over naive PPG is real; the individual rungs are not
  separable on two seasons** (2026-08-12, eval/ablation-report.md): `ppg → full`
  earns +187 (CI [+1.0, +8.7] per GW) and +238 (CI [+2.1, +10.3]), both excluding
  zero. Adjacent rungs mostly sit inside noise, and threshold-free greedy
  transfers flip sign by season (+44 / −46) — the maximin case for keeping the
  policy gate.

## OBSERVED FACTS (true, not yet validated as decision-changing)

- Bootstrap carries prior-season aggregates preseason; `form`=0 for everyone
  (2026-08-11).
- Salah is absent from the 2026/27 player list; new and promoted players have no
  history at all (2026-08-11).
- 2025/26 introduced defensive-contribution points; the 2024/25 archive lacks the
  column and the component model handles the absence gracefully (2026-08-11).
- Liverpool's GW15 2024/25 postponement appeared as a blank in fixtures data and
  the model zeroed ep_next correctly (2026-08-11).
- FPL publishes zero for every team strength rating in preseason; fixture
  difficulty is inert until they land (2026-08-12).
- Goalkeepers record no defensive-contribution actions in the live data (0 of 64)
  and the stat is absent from replay frames (2026-08-12).

## HYPOTHESES (untested — do NOT act on these)

- The hit threshold may be too loose in high-churn seasons (2026-08-11): the
  2023/24 sim took 68 hit points vs 52 in 2024/25 for slightly less engine gain.
  Test: rerun strategy_sim with HIT_GAIN_MIN 7-8 first.
- Betting-market implied probabilities could improve the team-level channels
  (2026-08-12). Shadow-mode only, and promotion needs a pre-registered forward
  test — team odds do not identify which player scores.
- Team optionality (cheap bench, price flexibility) has value the optimizer's
  bench weight does not capture (2026-08-11).
- The cross-regime PPG weight of 0.35 is a conservative guess (2026-08-12) and
  cannot be validated until the new BPS regime has real gameweeks. Revisit with
  the ablation harness.

## Process rules

- GW1 cold-start remains the weakest spot; mitigate with signals, never tune
  the model to fix a single famous gameweek (hindsight fitting).
- Rerun `uv run python eval/run_backtests.py` AND `eval/strategy_sim.py` after
  any model change; paste before/after aggregates here.
- Do not promote a hypothesis on a single season's evidence — the λ mistake above
  is what that looks like.
