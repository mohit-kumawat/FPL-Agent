"""Daily report writer: markdown for humans, JSON twin for the headless agent."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def _players_line(df: pd.DataFrame) -> str:
    return ", ".join(
        f"{r.web_name} ({r.team_short} {r.position} £{r.price}m)"
        for r in df.itertuples()
    )


def decision_summary(ctx: dict) -> dict:
    """Compact record of today's recommendation for decisions.jsonl."""
    rec = ctx.get("recommendation") or {}
    out = {"date": ctx["date"], "triggers": ctx["work"]["triggers"]}
    if "transfers" in rec:
        d = rec["transfers"]
        out["action"] = d["action"]
        if d["plan"] and d["plan"]["n_transfers"] > 0:
            out["out"] = list(d["plan"]["out"]["web_name"])
            out["in"] = list(d["plan"]["in"]["web_name"])
            out["net_gain"] = d["plan"]["net_gain_vs_hold"]
    if "initial_build" in rec:
        out["action"] = "initial_squad_proposal"
        out["squad"] = list(rec["initial_build"]["squad"]["web_name"])
        out["cost"] = rec["initial_build"]["cost"]
    if "captain" in rec:
        out["captain"] = rec["captain"]["pick"]
    return out


def _render_md(ctx: dict) -> str:
    changes, work = ctx["changes"], ctx["work"]
    rec = ctx.get("recommendation")
    lines = [f"# FPL Daily — {ctx['date']}", ""]

    # 0. where we are
    stage = ctx.get("stage")
    if stage:
        lines += ["## Stage"]
        dl = stage.get("hours_to_deadline")
        lines.append(f"- **{stage['stage']}** — GW{stage['next_gw']} deadline "
                     f"{stage.get('next_deadline')}"
                     + (f" ({dl:.0f}h)" if dl is not None else "")
                     + (" | team saved" if stage["has_team"] else " | **no team saved yet**"))
        for a in stage.get("stage_playbook", []):
            lines.append(f"- next: {a}")
        pend = stage.get("pending", [])
        if pend:
            lines.append("- **Pending items:**")
            for p in pend:
                due = f" (due {p['due']})" if p.get("due") else ""
                lines.append(f"  - [ ] {p['text']}{due}")
        lines.append("")

    # 0b. verification
    ver = ctx.get("verification")
    fresh = ctx.get("freshness")
    if ver:
        lines += ["## Verification"]
        if fresh:
            lines.append(f"- ✓ prices from official FPL API, fetched "
                         f"{fresh.get('fetched_at')} "
                         f"({'refetched this run' if fresh.get('refetched') else 'cached, post price-change'})")
        if ver["ok"]:
            for c in ver["checks"]:
                lines.append(f"- ✓ {c}")
        for w in ver.get("warnings", []):
            lines.append(f"- ⚠ {w}")
        for b in ver.get("blockers", []):
            lines.append(f"- ✗ **BLOCKED**: {b}")
        lines.append("")

    # 1. what changed
    lines += ["## What changed"]
    if changes["first_run"]:
        lines.append("- First run: baseline established.")
    for pc in changes["price_changes"][:12]:
        lines.append(f"- [DATA] Price: {pc['player']} £{pc['from']}m → £{pc['to']}m")
    if len(changes["price_changes"]) > 12:
        lines.append(f"- [DATA] …and {len(changes['price_changes']) - 12} more price moves")
    for sc in changes["status_changes"][:10]:
        lines.append(f"- [DATA] Status: {sc['player']} {sc['from']}→{sc['to']} — {sc['news'] or 'no detail'}")
    for nc in changes["news_changes"][:10]:
        lines.append(f"- [DATA] News: {nc['player']} — {nc['news']}")
    if not any([changes["price_changes"], changes["status_changes"],
                changes["news_changes"], changes["first_run"]]):
        lines.append("- Nothing material.")
    gs = changes["gw_state"]
    dl = work.get("hours_to_deadline")
    lines.append(f"- [DATA] Next: GW{gs['next_gw']} deadline {gs['next_deadline']}"
                 + (f" ({dl:.0f}h away)" if dl is not None else ""))

    # 2. models ran
    lines += ["", "## Models ran"]
    lines.append("- " + (", ".join(ctx["models_ran"]) if ctx["models_ran"]
                          else "None — " + "; ".join(work["triggers"])))

    # 3. key findings
    lines += ["", "## Key findings"]
    for f in ctx["findings"]:
        lines.append(f"- {f}")
    if ctx.get("rating"):
        r = ctx["rating"]
        lines.append(f"- [MODEL] Squad rating: **{r['overall_grade']}** "
                     f"({r['overall_pct']}% of optimal XI EP; gap {r['ep_gap']} EP)")
        for risk in r["risks"]:
            lines.append(f"- {risk}")
    for n in ctx.get("signal_notes", []):
        lines.append(f"- [SIGNAL] {n.get('date', '?')} {n.get('source', '?')}: {n.get('notes', '')}")
    if not ctx["findings"] and not ctx.get("rating") and not ctx.get("signal_notes"):
        lines.append("- None today.")

    # 4. recommended action
    lines += ["", "## Recommended action"]
    if rec is None:
        lines.append("- **No action.** " + "; ".join(work["triggers"]))
    else:
        if "initial_build" in rec:
            b = rec["initial_build"]
            lines.append(f"- [RECOMMENDATION] Initial squad (£{b['cost']}m):")
            lines.append(f"  - {_players_line(b['squad'])}")
        if "transfers" in rec:
            d = rec["transfers"]
            alt = d.get("best_alternative")
            if alt:
                lines.append(f"- **Net EP of best action vs hold: "
                             f"{alt['net_gain_vs_hold']:+.1f}** "
                             f"({alt['n_transfers']} transfer(s))")
            if d["action"] == "hold":
                lines.append("- [RECOMMENDATION] **Hold** — bank the free transfer.")
            else:
                p = d["plan"]
                lines.append(f"- [RECOMMENDATION] **{d['action']}** "
                             f"(net {p['net_gain_vs_hold']:+.1f} EP, hit {p['hit_cost']})")
                lines.append(f"  - OUT: {_players_line(p['out'])}")
                lines.append(f"  - IN: {_players_line(p['in'])}")
            for reason in d["reasoning"]:
                lines.append(f"  - {reason}")
        if "xi" in rec:
            xi = rec["xi"]
            lines.append(f"- [MODEL] Best XI ({xi['formation']}), "
                         f"EP {xi['expected_points']}: {_players_line(xi['xi'])}")
            lines.append(f"  - Bench order: {_players_line(xi['bench_order'])}")
        if "captain" in rec:
            c = rec["captain"]
            conf = "confident" if c["confident"] else f"thin margin ({c['margin']} EP)"
            eo = f", owned {c['ownership']:.0f}%" if c.get("ownership") is not None else ""
            lines.append(f"- [RECOMMENDATION] Captain **{c['pick']}**, vice {c['vice']} — {conf}{eo}.")
            if c.get("differential"):
                dd = c["differential"]
                lines.append(f"  - differential option: {dd['pick']} "
                             f"(EP {dd['ep_next']}, owned {dd['ownership']:.0f}%) — "
                             "for rank-chasing only")
        for note in rec.get("chips", []):
            lines.append(f"- {note}")
    if ctx.get("prev_decision"):
        pd_ = ctx["prev_decision"]
        lines.append(f"- [MEMORY] Previous decision ({pd_.get('date')}): "
                     f"{pd_.get('action')} — compare before acting.")

    # 4b. uncertainty flags
    unc = ctx.get("uncertainty") or []
    if unc:
        lines += ["", "## Uncertainty flags"]
        for u in unc:
            lines.append(f"- {u}")

    # 5. long-term outlook / 6. monitor
    lines += ["", "## Long-term outlook"]
    lines.append(f"- Planning horizon: next {config.HORIZON_GWS} GWs (EP figures above are horizon-weighted).")
    lines += ["", "## Monitor next"]
    monitors = []
    if changes["status_changes"] or changes["news_changes"]:
        monitors.append("follow-ups on today's injury news (press conferences)")
    if dl is not None and dl > 72:
        monitors.append("price change momentum on watchlist")
    if dl is not None and dl <= 72:
        monitors.append("late team news before deadline; rerun with --force after any flag")
    monitors.append("write expert notes to signals/*.yaml to nudge the model")
    for m in monitors:
        lines.append(f"- {m}")
    return "\n".join(lines) + "\n"


def _json_ctx(ctx: dict) -> dict:
    """JSON-safe subset of the run context."""
    def clean(o):
        if isinstance(o, pd.DataFrame):
            return o.to_dict("records")
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    keep = {k: ctx.get(k) for k in
            ("date", "stage", "verification", "freshness", "changes", "work",
             "models_ran", "findings", "signal_notes", "uncertainty")}
    rec = ctx.get("recommendation")
    if rec:
        keep["decision"] = decision_summary(ctx)
        if "captain" in rec:
            keep["captain"] = rec["captain"]
        if "xi" in rec:
            keep["xi"] = {
                "formation": rec["xi"]["formation"],
                "expected_points": rec["xi"]["expected_points"],
                "players": list(rec["xi"]["xi"]["web_name"]),
                "bench_order": list(rec["xi"]["bench_order"]["web_name"]),
            }
    if ctx.get("rating"):
        r = dict(ctx["rating"])
        r.pop("optimal_squad", None)
        keep["rating"] = r
    return clean(keep)


def write_report(ctx: dict) -> tuple[Path, Path]:
    md = config.REPORTS_DIR / f"{ctx['date']}.md"
    js = config.REPORTS_DIR / f"{ctx['date']}.json"
    md.write_text(_render_md(ctx))
    js.write_text(json.dumps(_json_ctx(ctx), indent=1, default=str))
    return md, js
