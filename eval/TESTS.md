# Historical backtest suite (2024/25) — specification

Ground rule: **no information published after the target GW's deadline.** The
replay harness enforces this structurally (only rows with round < GW enter the
snapshot; prices are the last observed before the GW). Compare predictions to
actuals afterward.

| # | GW(s) | Ask | Actual (for scoring) |
|---|-------|-----|----------------------|
| 1 | GW1→3 | Build best GW1 squad, premium decisions, long-term holds; replay GW2, GW3 | Haaland & Salah 41 pts each after GW3 (Haaland 7 goals) — agent should upgrade them to long-term holds on underlying data |
| 2 | GW1 | Captain among Haaland, Salah, Isak, Saka, Watkins | Salah 14, Saka 12, Haaland 7, Isak 5, Watkins 2 |
| 3 | GW6 | Captain + highest-upside players | Palmer 25 (4 goals vs Brighton) |
| 4 | GW10/11 | Restructure around Salah / Haaland / Palmer? 6-8 GW view | Salah 12 GI, 2 blanks, 9.3 pts/match; Haaland 11 GI, 4 blanks, 7.7; Palmer 12 GI, 8.2 |
| 5 | GW14 | Reassess premium structure — does the thesis update? | From GW14 on: Salah 211, Palmer 114, Haaland 95 |
| 6 | GW24 (DGW) | Triple Captain Salah? EV of chip vs alternatives | Salah 29 pts (BOU+EVE); 1.09M TCs = 87 pts. Judge reasoning, not outcome |
| 7 | GW25 | After GW24: keep/captain Salah again or change? Memory test | Salah 20 pts (WOL+AVL) |
| 8 | GW38 | Final-GW transfers/captain/chips | Salah 10, Bowen 13 (6 straight GWs with returns), Haaland 6, Palmer 3 |
| 9 | any | **No-change replay**: same data twice → second run must say "no material change, no models rerun" | determinism + trigger-matrix check |
| 10 | GW24 | **Hindsight protection**: prove the snapshot contains nothing from ≥ GW24 | structural assertion |

Scoring stance: judge decision quality given information at the time, not
outcome luck. Run across 10–20 GWs for aggregate metrics (captain hit rate,
top-11 actual points), not just the famous weeks.
