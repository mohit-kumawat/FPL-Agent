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
3. Write what you find as **minutes signals** — prefer the `role:` vocabulary
   (`expected_starter`, `rotation_risk`, `managed_minutes`, `bench_role`,
   `not_in_predicted_xi`, `ruled_out`) over raw bounds. **Each file needs an
   `evidence:` block** (tier, url, publisher, published_at) or it applies to
   nothing, and `expected_starter` / `ruled_out` need a tier-1 club source or
   two independent tier-2 outlets — one predicted lineup is a forecast, not a
   fact. Then `uv run fpl daily --force` once to merge them.
4. Compare the model's build against expert consensus where they disagree:
   `uv run fpl build` vs `uv run fpl build --lock <ids>`, present both EPs.
5. Preseason the model is at its weakest: team strength ratings may still be all
   zero (fixture difficulty then reads neutral, and the report says so), and
   prior-season scoring crosses the BPS rework, so cold-start uncertainty flags
   are expected rather than alarming. Say plainly which picks rest on a price
   prior rather than evidence. Bench Boost and Triple Captain are technically
   playable in GW1 — hold both unless the owner asks; the initial squad is the
   real wildcard.

The brief is the decision document. For every non-obvious pick, one line of
reasoning and its source. End with a **Falsifiers** section: for the captain
and any contested starter, state what late news would change the call
("captain X unless he's not in the leaked XI").

You are unattended: never wait for an answer. The squad is a **proposal** —
the owner records the go/no-go with `uv run fpl approve approved|rejected|
deferred`, and only then enters it on the FPL site and updates `squad.yaml`.
Never do either yourself. Use `uv run fpl pending add "<text>"` for
non-decision follow-ups.

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
<the go/no-go — `uv run fpl approve approved|rejected|deferred` — plus any
OWNER CHOICE the decision gate handed over, and anything else only the owner
can decide>
```
