# Part 2 — weekly recommendation

This prompt runs once squad.yaml has a team. `AGENT.md` is the system prompt;
don't repeat the runbook here.

---

Run the weekly loop and produce a recommendation **with its reasoning**.

**Hold is the default.** The backtest evidence (eval/agent-backtest-report.md)
is that over GW1-10 of the clean holdout season the transfer engine roughly
broke even after hits. A transfer must argue its way past that: recommend one
only when the report shows clear net EP gain, and say what the gain is.

1. Orient: `uv run fpl status`, then `uv run fpl daily`, read today's report.
2. Research only what the report's **Monitor next** flags. Minutes facts are
   the valuable output: injuries, suspensions, predicted lineups, press
   conferences. Write findings as minutes signals (`xmins_min`/`xmins_max`,
   with a source line), then `uv run fpl daily --force` once. Not writing a
   signal is the normal outcome of research.
3. In a teamnews-window run, check leaked/confirmed lineups against the
   current XI and captain falsifiers from the last brief, and say whether any
   falsifier fired.

You are unattended: never wait for an answer; record owner decisions with
`uv run fpl pending add "<text>"`. Never submit anything to the FPL site.

Element-id lookup when writing signals (never Read the snapshot, ~1.3MB one line):

```
jq -r '.elements[] | select(.web_name|test("Saka";"i")) | "\(.id) \(.web_name)"' \
  data/snapshots/bootstrap_*.json
```

Write the brief to the exact path given at the end of this message. Keep quiet
weeks to a few lines. Structure:

```markdown
# <date> — GW<n> <window>: <one-line verdict>

**Status:** quiet | action needed | blocked
**Deadline:** GW<n> in <n>h

## Recommendation
<hold / transfer X→Y / captain Z — with the net EP and the reasoning.
"Hold, nothing beats the squad" is a complete recommendation.>

## Why
<the evidence: model numbers + any researched facts, each with a source>

## Falsifiers
<what late news would change this — checked against in the teamnews run>

## Needs you
<only decisions the owner must make, or "nothing">
```
