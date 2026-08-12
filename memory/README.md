# memory/

The pipeline's persistent state. Plain files so a headless agent — or a human —
can read and write them without running Python.

Only `learnings.md` is tracked in git. Everything else is your own run history
and is gitignored.

Each scheduled run is a **fresh agent session with no conversation history**, so
files are the only continuity. That splits this directory into four lifecycles:

**Generated — the handoff.** Rewritten every run, bounded, never appended to.
This is the only season memory the next run is given.

| File | Purpose |
|---|---|
| `current-context.md` | The season in ~1KB: GW, squad, chips + expiry, points/rank, calibration trend, research accuracy, recent decisions. **Read this first.** |
| `digest.json` | The same, machine-readable |

**Ledgers — append-only evidence.** Never rewritten, so the audit trail holds.
Compacted into the handoff rather than read directly.

| File | Purpose |
|---|---|
| `decisions.jsonl` | Every recommendation |
| `runs.jsonl` | What ran when, and which triggers fired |
| `calibration.jsonl` | Per-GW model accuracy once points are final |
| `signal_log.jsonl` | Which minutes claims a GW's advice rested on, with source attribution |
| `signal_scores.jsonl` | Whether those claims held — the agent's own scorecard |
| `predictions/gwXX.csv` | Stored forecasts, scored after lockdown |

**State — small and mutable.** `state.json` (stage, GW state, signals seen,
transfer targets, scored GWs), `pending.json` (open items).

**Curated knowledge — the only hand-maintained files.** `learnings.md` (tracked;
three tiers, see below) and `research/*.md` (dated notes with sources).

## Retention

Nothing here grew without limit by accident — a 40-week season is ~280 runs, and
the two costs are disk and signal-to-noise. `fpl daily` enforces this every run
(`fpl_agent/retention.py`), and all of it is idempotent:

| Artifact | Policy | Why |
|---|---|---|
| `data/snapshots/` | keep 14 days + **every deadline day** | 1.4MB/day is ~790MB/season; deadline snapshots are the evidence of what was knowable when a decision was made, so they are kept forever |
| `signals/*.yaml` | expired files move to `signals/archive/` | expiry stopped them acting but not from emitting an "IGNORED" line into *every* future report |
| `routine/logs/*.log` | keep 30 days (ledger kept) | full agent stdout per run |
| `eval/phase3-predictions.jsonl` | brief sealed by hash + path, not embedded | the hash is what makes revision detectable; the copy was just bulk |

Safe to delete by hand: anything under `data/snapshots/`, `reports/`,
`routine/logs/`. Everything else is either regenerable or the audit trail.

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

Prose alone did not enforce that, so `fpl daily` now validates the file and flags
violations in the report: **every entry needs a `YYYY-MM-DD` date, VALIDATED
entries need an evidence pointer** (an eval report, a metric, or a gameweek), and
each tier is capped (12 / 20 / 15) so reaching the cap forces a prune or a
promotion instead of another append. A `## Process rules` section is exempt.
