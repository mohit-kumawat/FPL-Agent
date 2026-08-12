# routine/

The unattended layer: launchd runs a headless `claude` (following
[AGENT.md](../AGENT.md)) on a deadline-relative cadence, in two parts:

- **Part 1 — GW1 build** (`gw1_prompt.md`, active while `squad.yaml` has no
  team): research starters/roles/injuries where the model is blind, and produce
  the initial-squad decision document — the 15, XI, captain, bench, and the
  reasoning per non-obvious pick.
- **Part 2 — weekly loop** (`weekly_prompt.md`, active after): run the
  pipeline, research only what it flags, and produce a recommendation with its
  reasoning. **Hold is the default** — the backtest showed the transfer engine
  roughly breaks even after hits, so a transfer has to argue its way in.

The split follows the evidence in [eval/agent-backtest-report.md](../eval/agent-backtest-report.md):
~92% of the pipeline's GW1-10 edge came from the GW1 build, and agent
quality-opinions measured *negative* — so Part 1 is where effort goes, Part 2
defaults to doing nothing, and signals are minutes-facts only.

```
launchd (fixed slots; gate maps each wake-up to a window or a skip)
  └─ run.sh
       ├─ gate.py           which window, which mode — or skip (free, no tokens)
       ├─ claude --bare -p  the agent run (gw1 or weekly prompt)
       ├─ phase3 lock       hash-seals the recommendation (decision/teamnews)
       └─ build_view.py     regenerates view.html from all briefs
```

## The cadence

A gameweek has one deadline, so the runs cluster around it:

| Window | When (to deadline) | Job |
|---|---|---|
| `scan` | >30h out, at most every 72h | news sweep, write minutes signals |
| `decision` | 4–30h | the recommendation run |
| `teamnews` | 0.5–4h | check leaked lineups against the brief's falsifiers |

One `ok` run per (GW, `decision`/`teamnews`) window; scan repeats every ~3 days
so a three-week preseason gets several research passes, not one. Failures retry
at later slots, at most 3 attempts per window, then the window is given up — a
dead API key can't burn tokens at every slot forever. Inside 30 minutes of a
live deadline nothing fires: advice that can't be acted on isn't worth paying
for.

Dedup is recency-scoped, never forever — a `decision` run only counts against
this window for 36h, so last season's ledger can't silence this season. And a
deadline sitting in the *past* is treated as a stale state file, not a stop:
it classifies as `scan`, the run refreshes the state, and the routine rolls to
the next gameweek on its own.

## Setup (once)

1. **Credential** — a personal Anthropic API key in `.env` at the project root
   (gitignored):

   ```
   ANTHROPIC_API_KEY=sk-ant-api...
   ```

   The run uses `--bare`, which reads *only* this variable — it can't touch the
   keychain or any interactive Claude login.

2. **Install the schedule:** `./routine/install.sh`

3. **Prove it works without waiting:**

   ```bash
   FPL_ROUTINE_FORCE=1 ./routine/run.sh
   open routine/view.html
   ```

## Reading the output

- **`routine/view.html`** — one page, all briefs, newest first, status dot per
  run (green quiet / amber action needed / red blocked). Rebuilt after every
  run, works offline.
- **`memory/briefs/<date>-<window>.md`** — the same content as markdown.
- **`routine/logs/<date>.log`** — full agent output, when you want the *how*.
- **`eval/phase3-predictions.jsonl`** — every decision/teamnews recommendation,
  SHA-256-sealed at write time. `uv run python eval/phase3_prereg.py verify`
  proves none were edited after the fact; score them once GWs resolve.

## Guarantees

- **Never acts on your FPL account.** No cookie, no submit — decisions that
  need you land in the brief's "Needs you" and in `uv run fpl pending`.
- **A failed run is visible.** If the agent dies or writes no brief, the window
  gets a red "routine failed" brief pointing at the log. Silence always means
  "quiet", never "it broke" — a gate that crashes runs the agent anyway (only
  exit 78 is a real skip), and a missing prompt file aborts loudly rather than
  sending the agent an empty instruction.
- **Success means fresh output.** A run is scored on a brief written *after it
  started*, at the exact path it was given — a later window can't pass off an
  earlier brief as its own.
- **A good brief stays good.** If a later run dies, an earlier successful brief
  keeps its status and gains an amber "a later check failed" tag instead of
  being repainted red.
- **Recommendations are tamper-evident.** The hash lock means the forward test
  (the only uncontaminated evaluation this system can have) can't be quietly
  revised into looking right.
- **No double runs.** A lock directory (`.routine.lock`) serializes runs; stale
  locks from a crashed run are cleared automatically.
- **Tool surface is pinned.** `Bash` is limited to `uv run fpl`, `jq` (element-id
  lookup in the 1.3MB single-line snapshot), and a few read-only commands, plus
  Read/Write/Edit/Glob/Grep and WebSearch/WebFetch for research. No git, no
  package installs.

## Quirks worth knowing

- **Mac asleep at a slot?** launchd runs the missed job on wake (coalesced to
  one); the gate then maps it to whatever window is open. Powered off → the
  window may be missed entirely; the next window still runs.
- `claude --bare -p` reads the prompt from stdin — don't pass it as an argument
  (`--allowedTools` is variadic and would eat it).
- Everything this layer writes is gitignored: briefs, logs, `view.html`,
  `eval/phase3-predictions.jsonl` (it contains your actual squad plans).
- `FPL_ROUTINE_FORCE=1` bypasses the gate; `FPL_ROUTINE_SKIP_PREFLIGHT=1` skips
  the credential probe; `FPL_ROUTINE_SKIP_LOCK=1` skips the hash lock.
