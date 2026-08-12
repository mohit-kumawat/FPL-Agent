"""Phase 2 — score the blind-label captain test.

Three arms over the same gameweeks:
  pipeline  highest ep_next (no LLM)
  anon      agent picking from features with identities hidden -> recall impossible
  named     agent picking with names visible -> recall available

named minus anon is the value of knowing the season. It is the number that says
whether an agent-arm backtest on historical data measures skill or memory.

Usage: uv run python eval/phase2_score.py <season>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import replay  # noqa: E402

HERE = Path(__file__).parent


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
    cand = json.loads((HERE / f"phase2-candidates-{season}.json").read_text())
    picks = json.loads((HERE / "phase2-picks.json").read_text())

    rows, totals = [], {"pipeline": 0.0, "anon": 0.0, "named": 0.0, "best": 0.0}
    for gw in sorted(cand["gws"], key=int):
        g = cand["gws"][gw]
        act = replay.actual_points(season, int(gw))
        by_code = {c["code"]: n for c, n in zip(g["anon"], g["named"])}
        by_name = {n["name"]: n for n in g["named"]}

        pipe_id = g["pipeline_pick_id"]
        anon_row = by_code[picks["anon"][gw]]
        named_row = by_name[picks["named"][gw]]
        # ceiling: best captain available in the candidate set that week
        best = max(float(act.get(n["id"], 0)) for n in g["named"])

        vals = {
            "pipeline": float(act.get(pipe_id, 0)),
            "anon": float(act.get(anon_row["id"], 0)),
            "named": float(act.get(named_row["id"], 0)),
            "best": best,
        }
        for k, v in vals.items():
            totals[k] += v
        rows.append((gw, g["pipeline_pick_name"], anon_row["name"],
                     named_row["name"], vals))

    print(f"# Phase 2 — captain blind-label test, {season} GW1-10\n")
    print("| GW | pipeline | pts | anon arm | pts | named arm | pts | best available |")
    print("|---|---|---|---|---|---|---|---|")
    for gw, p, a, n, v in rows:
        print(f"| {gw} | {p.split()[-1]} | {v['pipeline']:.0f} | {a.split()[-1]} | "
              f"{v['anon']:.0f} | {n.split()[-1]} | {v['named']:.0f} | {v['best']:.0f} |")
    print(f"| **total** | | **{totals['pipeline']:.0f}** | | **{totals['anon']:.0f}** "
          f"| | **{totals['named']:.0f}** | **{totals['best']:.0f}** |")

    print(f"\n- captain points are doubled in FPL, so these totals count once; "
          f"the swing on a squad is 2x")
    print(f"- named - anon = {totals['named'] - totals['anon']:+.0f} "
          f"(the measured value of season recall)")
    print(f"- anon - pipeline = {totals['anon'] - totals['pipeline']:+.0f} "
          f"(feature judgement beyond argmax ep)")
    print(f"- ceiling gap (best - named) = {totals['best'] - totals['named']:+.0f}")


if __name__ == "__main__":
    main()
