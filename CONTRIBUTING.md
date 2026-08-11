# Contributing

## Setup

```bash
uv sync
uv run pytest -q
cp squad.example.yaml squad.yaml   # then fill in your own team
```

## The one rule that matters

**Any change to the models, fixtures, optimizer or policy must be accompanied by
before/after numbers from the evaluation suite.** "It feels better" is not
evidence — this project has already caught two of its own overfitting mistakes
that way.

```bash
uv run python eval/run_backtests.py    # per-GW metrics on 2024/25
uv run python eval/strategy_sim.py     # full-season strategy return
```

Paste the headline block (captain mean, capture ratio, top-11, Spearman ρ,
p10–p90 coverage) in your PR description, plus the strategy totals if you touched
anything that affects transfers.

Current baseline to beat (2024/25, GW6–24):

| Metric | Value |
|---|---|
| Captain actual | 10.21 pts/GW |
| Ceiling capture | 63% |
| Top-11 actual | 5.59 pts/player/GW |
| Spearman ρ (starters) | 0.364 |
| p10–p90 coverage | 0.83 (target 0.80) |
| Season strategy return | 2,232 (24/25) · 2,267 (23/24) |

## Things that will get a PR rejected

- **Hindsight fitting.** Tuning a constant so one famous gameweek comes out
  right. Single-season constants are how λ ended up at 0.7 before a two-season
  grid showed the ranking flips; see `memory/learnings.md`.
- **Breaking point-in-time integrity.** `replay.py` must never see data from on
  or after the target gameweek. The suite asserts this; don't weaken it.
- **Silent knob changes.** Constants live in `config.py` with a comment stating
  the evidence. No evidence, no change.
- **Committing personal data.** `squad.yaml`, `reports/`, `signals/*.yaml`,
  `memory/` run-state and `data/` caches are gitignored for a reason.

## Evidence discipline

`memory/learnings.md` has three tiers: **VALIDATED** (tested, may drive config),
**OBSERVED FACTS**, **HYPOTHESES** (never act on these). New findings enter as
hypotheses and get promoted only with eval-suite evidence. Keep it that way — it
is what stops an LLM operator from turning a hunch into folklore into a config
change.

## Code style

Match the surrounding code: type hints on public functions, docstrings that say
*why* rather than *what*, comments reserved for constraints the code can't show.
No new runtime dependencies without a clear justification.
