# signals/

The research inbox. An agent writes structured findings here after web research;
the models read them on every run. See the schema in [AGENT.md](../AGENT.md).

`signals/*.yaml` is **gitignored** — your notes may quote paywalled sources and
reveal your team plans. Only `*.example.yaml` files are tracked; see
`2026-08-11-gw1-research.example.yaml` for a real, worked example.

Discipline that keeps this useful:

- **Minutes news** (predicted XI, injury return, rotation) -> `xmins_min` /
  `xmins_max`. This is the signal type that earns its keep — write these.
- **Quality/role news** is a *last resort*: only a fact minutes can't express
  (gained/lost penalties, position change) -> `ep_per_gw`, with a direct quote
  in `source`, kept within ±0.5 (validation allows ±2; don't use the headroom).
  "He's better than the model thinks" is never a signal — the blind-label test
  in `eval/agent-backtest-report.md` measured that kind of opinion at negative
  value. Not writing a signal is the normal outcome of research.
- Set `confidence: high|medium|low` — it scales `ep_per_gw` by 1.0/0.6/0.3.
  A manager's own words are `high`; an aggregator's guess is `low`.
- Don't restate what the API already knows (`news`, `chance_of_playing`).
- Delete files when they go stale. Signals are re-read every run.
