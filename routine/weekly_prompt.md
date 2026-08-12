# Part 2 — weekly recommendation

This prompt runs once squad.yaml has a team. `AGENT.md` is the system prompt;
don't repeat the runbook here.

---

Run the weekly loop and produce a recommendation **with its reasoning**.

**Hold is the default.** The backtest evidence (eval/agent-backtest-report.md)
is that over GW1-10 of the clean holdout season the transfer engine roughly
broke even after hits. A transfer must argue its way past that: recommend one
only when the report shows clear net EP gain, and say what the gain is.

0. Read `memory/current-context.md` — the season in ~1KB, including any
   proposal awaiting the owner and the current research queue.
1. Orient: `uv run fpl status`, then `uv run fpl daily`, read today's report.
   In a **teamnews-window** run use `uv run fpl daily --force` instead — the
   pipeline dedupes per day, and the morning report predates the press
   conferences this run exists to catch.
2. Research what the report's **Decision gate** and **Monitor next** sections
   flag — the gate's `next research` lines are the priority queue, because
   each one is blocking an action. Minutes facts are the valuable output:
   injuries, suspensions, predicted lineups, press conferences. Write findings
   as minutes signals — prefer the `role:` vocabulary (`expected_starter`,
   `rotation_risk`, `managed_minutes`, `bench_role`, `not_in_predicted_xi`,
   `ruled_out`) over raw bounds, **each file carrying an `evidence:` block**
   (tier, url, publisher, published_at) or it applies to nothing — then
   `uv run fpl daily --force` once. Not writing a signal is the normal outcome
   of research.
2b. Respect the gate verdict. **BLOCKED** means do the named research, not
   argue past it; if the research doesn't land before the deadline, the brief
   says so and recommends holding. **OWNER CHOICE** means present both paths
   with their risks and let the owner pick — never resolve it yourself.
3. In a teamnews-window run, check leaked/confirmed lineups against the
   current XI and captain falsifiers from the last brief, and say whether any
   falsifier fired.
4. After a gameweek has played, remember points are not final until 09:00 UK the
   next morning. A `calibration deferred` finding is expected and self-healing —
   don't flag it as a problem, and don't quote last gameweek's score or bonus as
   settled before then. If a chip window is about to close (first-half copies
   expire 2 January), say so under "Needs you".

You are unattended: never wait for an answer. A recommendation is a **proposal**
— the pipeline logs it for the owner to decide with `uv run fpl approve
approved|rejected|deferred`; say so in "Needs you" rather than implying it is
ready to execute. Note any still-pending or rejected proposal from the context
file instead of silently re-arguing it. Use `uv run fpl pending add "<text>"`
for non-decision follow-ups. Never submit anything to the FPL site, and never
edit `squad.yaml` on the owner's behalf.

Element-id lookup when writing signals (never Read the snapshot, ~1.3MB one line):

```
jq -r '.elements[] | select(.web_name|test("Saka";"i")) | "\(.id) \(.web_name)"' \
  data/snapshots/bootstrap_*.json
```

Write the brief to the exact path given at the end of this message. Keep quiet
weeks to a few lines. Structure:

```markdown
# <date> — GW<n> <window>: <one-line verdict>

**Status:** quiet | action needed | blocked | owner choice
**Deadline:** GW<n> in <n>h
**Gate:** QUALIFIED | BLOCKED | OWNER CHOICE

## Recommendation
<hold / transfer X→Y / captain Z — with the net EP and the reasoning.
"Hold, nothing beats the squad" is a complete recommendation. If the gate
blocked, the recommendation is the abstention: say what is missing.>

## Why
<the evidence: model numbers ([MODEL], forecasts) kept separate from
researched facts ([SIGNAL], each with its source and tier)>

## Claims used / rejected
<which signals applied and at what tier; which were rejected or in conflict,
and why — one line each, or "none">

## Falsifiers
<what late news would change this — checked against in the teamnews run>

## Needs you
<owner decisions: `uv run fpl approve approved|rejected|deferred`, any
OWNER CHOICE the pipeline handed over, use-it-or-lose-it chips — or "nothing">

## Next research
<the gate's unblocking tasks, most deadline-urgent first, or "none">
```
