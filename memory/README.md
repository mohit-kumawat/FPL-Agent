# memory/

The pipeline's persistent state. Plain files so a headless agent — or a human —
can read and write them without running Python.

Only `learnings.md` is tracked in git. Everything else is your own run history
and is gitignored: `state.json`, `pending.json`, `decisions.jsonl`, `runs.jsonl`,
`predictions/`, `research/`.

| File | Written by | Purpose |
|---|---|---|
| `learnings.md` | agent + humans | Three-tier evidence log (see below) — **tracked** |
| `state.json` | pipeline | Current beliefs: stage, GW state, signals seen |
| `pending.json` | pipeline + `fpl pending` | Open action items |
| `decisions.jsonl` | pipeline | Every recommendation; each report diffs against the last |
| `runs.jsonl` | pipeline | What ran when, and which triggers fired |
| `predictions/gwXX.csv` | pipeline | Stored forecasts, auto-scored after the GW |
| `research/*.md` | agent | Dated research notes with sources and trust calls |

## The three tiers in learnings.md

**VALIDATED** — tested against historical data via the eval suite. Only these
may justify a config change.
**OBSERVED FACTS** — true statements about the data or season, not yet shown to
change any decision.
**HYPOTHESES** — plausible ideas. The agent must never act on these; they are
prompts for future testing.

This separation exists because an LLM writing "lessons" into a file will
otherwise turn a hunch into folklore and folklore into a model constant. New
findings enter as hypotheses and are promoted only with before/after numbers.
