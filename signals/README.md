# signals/

The research inbox. An agent writes structured findings here after web research;
the models read them on every run. See the schema in [AGENT.md](../AGENT.md).

`signals/*.yaml` is **gitignored** — your notes may quote paywalled sources and
reveal your team plans. Only `*.example.yaml` files are tracked; see
`2026-08-11-gw1-research.example.yaml` for a real, worked example.

Discipline that keeps this useful:

- **Minutes news** (predicted XI, injury return, rotation) -> `xmins_min` / `xmins_max`
- **Quality/role news** that minutes can't express (gained penalties, moved
  forward) -> `ep_per_gw`, kept within [-2, 2]
- Set `confidence: high|medium|low` — it scales `ep_per_gw` by 1.0/0.6/0.3.
  A manager's own words are `high`; an aggregator's guess is `low`.
- Don't restate what the API already knows (`news`, `chance_of_playing`).
- Delete files when they go stale. Signals are re-read every run.
