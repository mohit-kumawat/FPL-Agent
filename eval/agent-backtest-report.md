# GW1-10 no-leak backtest

Seed 20260812. Pipeline arm only — no LLM in the loop (see report for why).

## Phase 0 — leak audit

### 2023-24
- data files: all present
- team strengths: preseason snapshot (played/points/position all zero) — clean
- truncation equivalence GW1-10: PASS (all identical)

### 2024-25
- data files: all present
- team strengths: **NOT preseason** — played=[0] points=[0] position_populated=True
- truncation equivalence GW1-10: PASS (all identical)

## Phase 1 — pipeline arm vs benchmarks

> **Correction (2026-08-12):** the template arm originally filtered on position
> `"GKP"` while the raw merged rows use `"GK"`, so the crowd XI was missing its
> goalkeeper AND took its captain from the wrong row. First-published templates
> (495 / 548) were understated. Tables below use the fixed arm; the original
> conclusion "the pipeline beats the crowd" does **not** survive the fix — see
> Sample-size reality.

### 2023-24
| arm | GW1-10 total | detail |
|---|---|---|
| pipeline (transfers) | **637** | 12 moves, 12 hit pts, raw 649 |
| pipeline (hold GW1) | 626 | no transfers |
| template (most-owned XI) | 563 | crowd benchmark |
| random legal squad | 145 | null: p50 139, p90 211, n=200 |
| ceiling (perfect XI from held squad) | 745 | selection upper bound |

- per-GW (pipeline): [58, 35, 44, 74, 70, 86, 89, 58, 72, 63]
- per-GW (template): [71, 43, 39, 74, 41, 71, 47, 33, 73, 71]

### 2024-25
| arm | GW1-10 total | detail |
|---|---|---|
| pipeline (transfers) | **576** | 13 moves, 16 hit pts, raw 592 |
| pipeline (hold GW1) | 536 | no transfers |
| template (most-owned XI) | 626 | crowd benchmark |
| random legal squad | 150 | null: p50 144, p90 210, n=200 |
| ceiling (perfect XI from held squad) | 708 | selection upper bound |

- per-GW (pipeline): [59, 72, 89, 33, 54, 50, 49, 53, 75, 58]
- per-GW (template): [85, 86, 87, 59, 65, 63, 47, 30, 67, 37]


## Phase 0c — contamination probe (scored)

Claims written in `eval/contamination-probe.md` *before* any data was queried,
then scored:

| Claim | Actual | Verdict |
|---|---|---|
| Haaland ≈ £14.0m at GW1 | £14.0m | exact |
| Salah ≈ £12.5m at GW1 | £12.5m | exact |
| Son hat-trick vs Burnley | GW4, 3 goals, 20 pts | exact, incl. gameweek |
| Newcastle 8-0 Sheffield Utd | GW6, 0-8 | exact, incl. scoreline |
| Ferguson hat-trick vs Newcastle | GW4, 3 goals, 17 pts | exact, incl. gameweek |
| Promoted: Luton, Burnley, Sheffield Utd | all three present | correct |
| Trippier assist-heavy standout | 6 assists, 59 pts in 10 GWs | correct |
| Palmer's scoring came after GW10 | 29 pts across GW1-10 | correct |

**8/8 checkable claims correct, five of them exact to the gameweek, the price
decimal, or the scoreline.** Contamination is total. Any agent-arm result on
2023-24 or 2024-25 measures recall, not skill, and Phase 2 is therefore not
worth running on historical data.

## Sample-size reality

Pipeline minus template, per gameweek:

| Season | mean/GW | sd | 95% CI (per GW) | 10-GW total CI | GWs won |
|---|---|---|---|---|---|
| 2023-24 (clean) | +8.6 | 18.4 | [-1.7, +19.8] | [-17, +198] | 5/10 |
| 2024-25 (tuned-on) | -3.4 | 17.5 | [-13.3, +6.9] | [-133, +69] | 5/10 |

**Both intervals cross zero** against the corrected template. The honest
GW1-10 conclusion: the pipeline clearly beats random and holds its own against
the crowd, but a crowd edge is NOT demonstrated — on the (contaminated) second
season it trails the template outright. The hold-vs-transfers comparison
(637 vs 626) is unaffected by this correction.

At sd ≈ 20 pts/GW, resolving a 5 pts/GW edge needs **~61 gameweeks**. Ten
gameweeks cannot separate skill from variance at this effect size.

## Phase 2 — captain blind-label test (executed)

| GW | pipeline | pts | anon arm | pts | named arm | pts | best available |
|---|---|---|---|---|---|---|---|
| 1 | Haaland | 13 | Haaland | 13 | Haaland | 13 | 13 |
| 2 | Disasi | 0 | Haaland | 2 | Haaland | 2 | 11 |
| 3 | Haaland | 4 | Haaland | 4 | Haaland | 4 | 8 |
| 4 | Haaland | 20 | Haaland | 20 | Haaland | 20 | 20 |
| 5 | Haaland | 6 | Haaland | 6 | Haaland | 6 | 10 |
| 6 | Haaland | 6 | Haaland | 6 | Trippier | 18 | 18 |
| 7 | Morris | 10 | Haaland | 2 | Haaland | 2 | 17 |
| 8 | Salah | 15 | Salah | 15 | Salah | 15 | 15 |
| 9 | Salah | 16 | Salah | 16 | Salah | 16 | 16 |
| 10 | Salah | 8 | Salah | 8 | Salah | 8 | 16 |
| **total** | | **98** | | **92** | | **104** | **144** |

- captain points are doubled in FPL, so these totals count once; the swing on a squad is 2x
- named - anon = +12 (the measured value of season recall)
- anon - pipeline = -6 (feature judgement beyond argmax ep)
- ceiling gap (best - named) = +40

Design: identical pre-deadline rows shown two ways. The **anon** arm sees
features with player and team identity replaced by per-GW salted codes. The **named** arm sees names, and recall was
deliberately used once (GW6) to price it. Picks were recorded in
`eval/phase2-picks.json` before any actual points were read.

Captain choice adds one extra copy of the player's score, so these deltas
are the points swing on a squad directly.

**Methodology caveat (added after review):** the anonymization is weaker than
"recall is impossible" implies. Anon and named entries share order and carry
identical feature values in one file, so the mapping is recoverable by index or
by exact price/EP match — and the generator printed both views adjacently. The
picks were recorded before actuals were read, and the anon arm's picks were
made from features alone, but the artifact cannot *prove* that. Treat -6/+12 as
one indicative 10-GW observation, not a controlled measurement. The generator
now emits a separate shuffled anon view for any future run.

## Phase 3 — forward test (locked, not yet scorable)

GW1 2026-27 prediction sealed before the deadline:

```
  [1] GW1 locked 2026-08-12T05:34:53+00:00 OK
all predictions intact
```

2026-27 is past the model's knowledge cutoff, so this is the only arm with
no contamination. `verify` recomputes the SHA-256 of every stored
prediction — an edit after the fact voids the result rather than silently
improving it (tested: editing the payload flips it to TAMPERED).
