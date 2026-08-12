# signals/

The research inbox. An agent writes structured findings here after web research;
the models read them on every run. See the schema in [AGENT.md](../AGENT.md).

`signals/*.yaml` is **gitignored** — your notes may quote paywalled sources and
reveal your team plans. Only `*.example.yaml` files are tracked; see
`2026-08-11-gw1-research.example.yaml` for a real, worked example.

Discipline that keeps this useful:

- **Every file needs an `evidence:` block** — `tier`, `url`, `publisher`,
  `published_at`. Without one (or with a tier the metadata can't back up) the
  file is a *watch item*: parsed, reported, applied to nothing. A source that
  can't be checked can't establish anything.
- **Minutes news** (predicted XI, injury return, rotation) -> `role:` vocabulary
  or `xmins_min` / `xmins_max`. This is the signal type that earns its keep.
- **Quality/role news** is a *last resort*: only a fact minutes can't express
  (gained/lost penalties, position change) -> `ep_per_gw`, with a direct quote
  in `source`, kept within ±0.5 (validation allows ±2; don't use the headroom).
  "He's better than the model thinks" is never a signal — the blind-label test
  in `eval/agent-backtest-report.md` measured that kind of opinion at negative
  value. Not writing a signal is the normal outcome of research.
- Set `confidence: high|medium|low` — it scales `ep_per_gw` by 1.0/0.6/0.3.
  A manager's own words are `high`; an aggregator's guess is `low`.
- Don't restate what the API already knows (`news`, `chance_of_playing`).
- **Don't delete stale files** — `ttl_days` / `expires` retires them and the
  pipeline moves expired ones to `signals/archive/` on its own.

## What each tier may do

| Tier | Source | May establish |
|---|---|---|
| 1 | club, manager, league, competition | anything, including `ruled_out` / `expected_starter` |
| 2 | named journalist or established outlet | forecast bounds (`rotation_risk`, `managed_minutes`, `bench_role`, `not_in_predicted_xi`) and `ep_per_gw`; hard availability roles only with a **second tier-2 file from a different domain** |
| 3 | aggregator, anonymous, or unverifiable | nothing — watch item only |

Tier 0 is the FPL API itself; a YAML file can't claim it (the pipeline already
reads the API, so a hand-copy would only add a place to be wrong).

A source is current for 14 / 7 / 3 days by tier, and expiry is the *earlier* of
that and your own `ttl_days`. `scenarios:` are evidence-gated too — they feed
chip EV, so tier 3 supplies none.

When two files contradict each other (a floor above a cap for one player), the
higher tier wins; equal tiers drop both bounds and the report asks for better
evidence. Check the report's **Key findings** and **Decision gate** sections
after writing a signal.
