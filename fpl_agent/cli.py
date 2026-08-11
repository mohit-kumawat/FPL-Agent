"""CLI: uv run fpl <command>

  daily      run the daily agent loop (change detection -> models -> report)
  build      build an optimal 15 from scratch and print it
  rate       rate the squad in squad.yaml
  backtest   validate models on the prior season
  refresh    force-refresh API snapshots
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="fpl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daily");    d.add_argument("--force", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("pending")
    p.add_argument("op", choices=["list", "add", "done"], nargs="?", default="list")
    p.add_argument("text", nargs="*")
    b = sub.add_parser("build");    b.add_argument("--budget", type=float, default=100.0)
    b.add_argument("--lock", type=str, default="",
                   help="comma-separated player ids to force into the squad")
    sub.add_parser("rate")
    bt = sub.add_parser("backtest"); bt.add_argument("--train-gws", type=int, default=26)
    sub.add_parser("refresh")
    args = ap.parse_args()

    from . import data

    if args.cmd == "refresh":
        data.refresh(force=True)
        print("snapshots refreshed")
        return

    if args.cmd == "status":
        import json as _json
        from . import lifecycle
        boot = data.refresh()["bootstrap"]
        print(_json.dumps(lifecycle.status(boot), indent=2, default=str))
        return

    if args.cmd == "pending":
        from . import lifecycle
        text = " ".join(args.text)
        if args.op == "add" and text:
            lifecycle.add_pending(text)
        elif args.op == "done" and text:
            print("marked" if lifecycle.complete_pending(text) else "no match")
        for i in lifecycle.load_pending():
            mark = "x" if i.get("done") else " "
            due = f" (due {i['due']})" if i.get("due") else ""
            print(f"[{mark}] {i['text']}{due}")
        return

    if args.cmd == "daily":
        from . import daily
        ctx = daily.run_daily(force=args.force)
        if "skipped" in ctx:
            print(ctx["skipped"])
            return
        print(f"triggers : {'; '.join(ctx['work']['triggers'])}")
        print(f"models   : {', '.join(ctx['models_ran']) or 'none'}")
        print(f"report   : {ctx['report_md']}")
        return

    from . import features, memoryio, models
    refreshed = data.refresh()
    boot, fixtures = refreshed["bootstrap"], refreshed["fixtures"]
    players = data.players_frame(boot)
    panel = data.build_current_panel(boot)
    enriched = features.enrich_players(players, boot, fixtures,
                                       panel if not panel.empty else None)
    signal_adjust, _ = memoryio.load_signals()
    ep = models.expected_points(enriched, panel if not panel.empty else None,
                                signal_adjust if len(signal_adjust) else None)

    if args.cmd == "build":
        from . import optimizer
        locked = [int(x) for x in args.lock.split(",") if x.strip()] if args.lock else None
        res = optimizer.build_squad(ep, budget=args.budget, locked=locked)
        xi = optimizer.pick_xi(res["squad"])
        print(f"cost £{res['cost']}m | formation {xi['formation']} | "
              f"EP(next, XI+C) {xi['expected_points']}")
        cols = ["web_name", "team_short", "position", "price", "ep_next", "ep_horizon"]
        print(xi["xi"][cols].to_string(index=False))
        print("--- bench ---")
        print(xi["bench_order"][cols].to_string(index=False))
        print(f"captain: {xi['captain']['web_name']} | vice: {xi['vice']['web_name']}")
        return

    if args.cmd == "rate":
        from . import memoryio, rating
        squad = memoryio.load_squad()
        ids = [p["id"] for p in (squad.get("players") or [])]
        if len(ids) != 15:
            sys.exit("squad.yaml has no 15-player squad yet — run `fpl build` first")
        r = rating.rate_squad(ep, ids)
        print(f"Squad grade {r['overall_grade']} ({r['overall_pct']}% of optimal, "
              f"gap {r['ep_gap']} EP)")
        for g in r["player_grades"]:
            print(f"  {g['grade']:>2}  {g['player']:<20} {g['pos']} £{g['price']}m "
                  f"ep{g['ep']} ({g['vs_best_in_bracket']}% of bracket best)")
        for risk in r["risks"]:
            print(f"  ! {risk}")
        return

    if args.cmd == "backtest":
        from . import backtest
        backtest.run(train_gws=args.train_gws)


if __name__ == "__main__":
    main()
