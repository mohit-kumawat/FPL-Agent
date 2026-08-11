# Learnings

Three tiers, per the evidence-discipline rule. **Only VALIDATED entries may
change model config.** Observed facts await more data; hypotheses are folklore
until tested — the agent must not act on them.

## VALIDATED (tested across historical data; may drive config)

- **Component-based EP beats PPG-compression** (2026-08-11). Same 2024/25
  GW6–24 sweep, same information: captain actual 9.42 → 10.00 pts/GW, ceiling
  capture 59% → 62%, top-11 5.57 → 5.60. Also resolved the GW1 Watkins-over-Salah
  miss without outcome tuning (Saka/Salah ranked 1-2, actual 12/14). Basis of
  W_COMPONENT=0.55.
- **No ensemble extreme is robust across seasons** (2026-08-11). λ grid on
  2024/25 AND 2025/26 test windows: rankings flip completely (A best/B worst,
  then B best/A worst). λ=0.6 is the maximin (worst-case top-11 4.14).
  The earlier λ=0.7 choice was overfit to one season — a caught mistake, kept
  here as a warning.
- **Strategy return, two full seasons**: 2024/25 = 2,284 pts (43 transfers,
  32 hit pts), +399 vs hold; 2023/24 = 2,195 pts (53 transfers, 72 hit pts),
  +251 vs hold. No chips/autosubs in either arm. Both seasons comfortably above
  the overall average (~2,030-2,100); below top-10k pace (~2,450) — honest gap.
- **Early-season data thresholds must self-scale** — fixed 900-minute history
  gate produced all-NaN EP at GW9/10 (caught by replay suite; fixed).

- **Logistic appearance points** (P(play) + logistic P(60')) over hard clip:
  captain 10.00 → 10.21, capture 62% → 63% on the 2024/25 sweep (2026-08-11).
- **ep_sd calibration**: uncalibrated p10-p90 covered 68% of top-50 actuals vs
  80% target → scaled sd ×1.3 → coverage 0.83. Metrics baseline for future
  model changes: rho=0.364 (starters), by position GKP .38/MID .35/DEF .34/FWD .24.

## OBSERVED FACTS (true, not yet validated as decision-changing)

- Bootstrap carries prior-season aggregates preseason; `form`=0 for everyone.
- Salah absent from the 2026/27 player list; new/promoted players have no history.
- 2025/26 introduced defensive-contribution points; 2024/25 archive lacks the
  column (component model handles absence gracefully).
- Liverpool's GW15 2024/25 fixture postponement appeared as a blank in fixtures
  data — the model zeroed ep_next correctly.

## HYPOTHESES (untested — do NOT act on these)

- The hit threshold may be too loose in high-churn seasons: 2023/24 sim took
  72 hit points vs 32 in 2024/25 for less transfer-engine gain (+251 vs +399).
  Test: rerun strategy_sim with HIT_GAIN_MIN 7-8 before changing config.

- Variance-seeking captaincy ("differential mode") may beat EP-max when chasing
  rank late-season. Needs a rank-simulation study before use.
- Betting-market implied probabilities could improve the attack channel.
- Team optionality (cheap bench, price flexibility) has value the optimizer's
  0.1 bench weight doesn't capture.

## Process rules

- GW1 cold-start remains the weakest spot; mitigate with signals, never tune
  the model to fix a single famous GW (hindsight fitting).
- Rerun `uv run python eval/run_backtests.py` AND `eval/strategy_sim.py` after
  any model change; paste before/after aggregates here.
