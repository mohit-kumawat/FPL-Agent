# Part 1 — GW1 squad and lineup

This prompt runs until squad.yaml has a team. `AGENT.md` is the system prompt;
don't repeat the runbook here.

---

Produce the GW1 decision document: the 15, the starting XI, captain, vice, and
bench order — for the coming weeks, not just GW1 (the optimizer already plans
over its horizon; don't stretch reasoning past it).

1. Orient: `uv run fpl status`, then `uv run fpl daily` and read today's report.
2. Research where the model is blind. Preseason it has no current-season data,
   so the highest-value facts are: confirmed starters, new-signing roles,
   penalty takers, injuries, and pre-season minutes. Web research is your job
   here — press conferences, predicted lineups, beat writers.
3. Write what you find as **minutes signals** (`xmins_min`/`xmins_max` with a
   source). Then `uv run fpl daily --force` once to merge them.
4. Compare the model's build against expert consensus where they disagree:
   `uv run fpl build` vs `uv run fpl build --lock <ids>`, present both EPs.

The brief is the decision document. For every non-obvious pick, one line of
reasoning and its source. End with a **Falsifiers** section: for the captain
and any contested starter, state what late news would change the call
("captain X unless he's not in the leaked XI").

You are unattended: never wait for an answer; record owner decisions with
`uv run fpl pending add "<text>"`. Never submit anything to the FPL site.

Signals need FPL element ids, and the reports don't carry them. Use `jq`
against the newest snapshot — never Read it, it's ~1.3MB on one line:

```
jq -r '.elements[] | select(.web_name|test("Saka";"i")) | "\(.id) \(.web_name)"' \
  data/snapshots/bootstrap_*.json
```

Never guess an id — a wrong one silently adjusts a different player.

Write the brief to the exact path given at the end of this message. Structure:

```markdown
# <date> — GW1 build: <one-line verdict>

**Status:** action needed
**Deadline:** GW1 in <n>h

## The squad
<the 15 with prices, XI marked, captain/vice, bench order>

## Why
<per non-obvious pick: reason + source. Skip the obvious ones.>

## Model vs consensus
<where they disagreed and which build you recommend>

## Falsifiers
<what news changes which call>

## Needs you
<the go/no-go and anything else only the owner can decide>
```
