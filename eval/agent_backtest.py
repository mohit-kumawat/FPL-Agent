"""No-leak GW1-10 backtest: Phase 0 (leak audit) + Phase 1 (pipeline arm).

Usage: uv run python eval/agent_backtest.py [season ...]   (default: 2023-24 2024-25)

Phase 0 proves the replay harness cannot see the future before any number is
believed. The existing assertion in replay.snapshot_at is a tautology -- it
filters to `round < gw` and then asserts max(round) < gw -- so the real test
here is truncation equivalence: physically delete every row from GW n onward,
recompute, and require byte-identical expected points. If deleting the future
changes nothing, nothing read the future.

Phase 1 runs the pipeline arm (no LLM) and scores it against benchmarks built
from the same dataset, so every comparison is apples to apples.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import season_loop  # noqa: E402
from fpl_agent import config, data, replay  # noqa: E402
from season_loop import run_season  # noqa: E402

OUT: list[str] = []
SEED = 20260812
GWS = 10


def log(s: str = "") -> None:
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------- phase 0
def data_completeness(season: str) -> dict:
    """Missing fixtures/teams degrade replay silently rather than raising."""
    root = Path(config.ROOT) / "data" / "history"
    have = {
        "merged_gw": (root / f"merged_gw_{season}.parquet").exists(),
        "fixtures": (root / f"fixtures_{season}.parquet").exists(),
        "teams": (root / f"teams_{season}.parquet").exists(),
    }
    return have


def strength_provenance(season: str) -> dict:
    """Team strengths are a single static snapshot with no time dimension.

    If it was captured mid- or post-season it encodes how good teams turned out
    to be, which is hindsight in every GW's fixture-difficulty term. A preseason
    capture has played/points/position all zero.
    """
    try:
        t = data.load_season_teams(season)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "why": f"unreadable: {exc}"}
    played = set(pd.to_numeric(t.get("played", pd.Series([0])), errors="coerce").fillna(0))
    points = set(pd.to_numeric(t.get("points", pd.Series([0])), errors="coerce").fillna(0))
    pos = set(pd.to_numeric(t.get("position", pd.Series([0])), errors="coerce").fillna(0))
    preseason = played <= {0} and points <= {0} and pos <= {0}
    return {"ok": preseason, "played": sorted(played)[:3],
            "points": sorted(points)[:3], "position_populated": pos != {0}}


def leak_audit(season: str, gws: int = GWS) -> list[dict]:
    """Truncation equivalence: deleting GW>=n data must not change the GW n view."""
    results = []
    original = replay._season_gws

    for gw in range(1, gws + 1):
        full = replay.expected_points_at(season, gw)

        # GW1 can't be truncated -- price legitimately comes from the round-1
        # row, and the stat columns are prior-season aggregates (a different
        # file, so untouched here). Zero every target-season stat instead: if
        # the view is unchanged, GW1 read identity and price and nothing else.
        keep = {"element", "name", "position", "team", "value", "round", "kickoff_time"}

        def truncated(s: str, _gw: int = gw, _keep: set = keep) -> pd.DataFrame:
            df = original(s)
            if s != season:
                return df
            if _gw > 1:
                return df[df["round"] < _gw]
            d = df.copy()
            for col in d.columns:
                if col not in _keep and pd.api.types.is_numeric_dtype(d[col]):
                    d[col] = 0
            return d

        # Caches must not straddle the patch in either direction: a pre-patch
        # frame served to patched code would make a genuine leak look clean.
        season_loop.clear_caches()
        replay._season_gws = truncated
        try:
            cut = replay.expected_points_at(season, gw)
        finally:
            replay._season_gws = original
            season_loop.clear_caches()

        cols = [c for c in ("id", "ep_next", "ep_horizon", "xmins", "price")
                if c in full.columns and c in cut.columns]
        a = full[cols].sort_values("id").reset_index(drop=True)
        b = cut[cols].sort_values("id").reset_index(drop=True)
        same = a.shape == b.shape and bool(
            (a.select_dtypes("number").round(9)
             == b.select_dtypes("number").round(9)).all().all()
        )
        delta = "identical" if same else f"DIFFERS rows {a.shape[0]}->{b.shape[0]}"
        results.append({"gw": gw,
                        "method": "zeroed-stats" if gw == 1 else "truncation-equivalence",
                        "clean": same, "detail": delta})
    return results


# ----------------------------------------------------------------- phase 1
def run_arm(season: str, max_gw: int = GWS, transfers: bool = True) -> dict:
    """The pipeline arm, through the shared eval season loop (eval/season_loop.py)."""
    r = run_season(season, max_gw=max_gw, transfers=transfers)
    # selection ceiling: the perfect XI from the squad actually held each GW
    ceiling = sum(best_xi_from(sq, replay.actual_points(season, gw + 1))
                  for gw, sq in enumerate(r["squads"]))
    return {"total": r["total_raw"], "raw": round(sum(r["weekly_raw"])),
            "hits": r["hits"], "moves": r["moves"],
            "weekly": [round(w) for w in r["weekly_raw"]],
            "autosub_total": r["total_autosub"],
            "ceiling": round(ceiling)}


def _valid_xi(rows: list[tuple[int, str, float]]) -> float:
    """Best-scoring legal XI from (id, pos, points); captain = top scorer."""
    by = {p: sorted([r for r in rows if r[1] == p], key=lambda r: -r[2])
          for p in ("GKP", "DEF", "MID", "FWD")}
    best = -1.0
    for ndef in range(3, 6):
        for nmid in range(2, 6):
            nfwd = 11 - 1 - ndef - nmid
            if not 1 <= nfwd <= 3:
                continue
            if len(by["GKP"]) < 1 or len(by["DEF"]) < ndef or \
               len(by["MID"]) < nmid or len(by["FWD"]) < nfwd:
                continue
            pick = by["GKP"][:1] + by["DEF"][:ndef] + by["MID"][:nmid] + by["FWD"][:nfwd]
            tot = sum(r[2] for r in pick) + max(r[2] for r in pick)   # captain
            best = max(best, tot)
    return max(best, 0.0)


def best_xi_from(squad_ep: pd.DataFrame, act: pd.Series) -> float:
    """Perfect XI + captain from the squad actually held -- the selection ceiling."""
    rows = [(int(r.id), str(r.position), float(act.get(int(r.id), 0)))
            for r in squad_ep.itertuples()]
    return _valid_xi(rows)


def template_arm(season: str, max_gw: int = GWS) -> dict:
    """What the crowd owned: most-selected legal XI each GW, most-selected captain.

    Raw merged rows label keepers "GK" while the replay layers normalize to
    "GKP" -- normalize here too, or the crowd XI silently fields ten men. The
    captain is the most-owned player IN the XI, not whoever pd.concat put
    first (that used to be the goalkeeper row).
    """
    df = replay._season_gws(season)
    df = df.assign(position=df["position"].replace({"GK": "GKP"}))
    weekly = []
    for gw in range(1, max_gw + 1):
        g = df[df["round"] == gw].copy()
        if g.empty:
            weekly.append(0.0); continue
        g["selected"] = pd.to_numeric(g["selected"], errors="coerce").fillna(0)
        g = g.sort_values("selected", ascending=False)
        by = {p: g[g["position"] == p] for p in ("GKP", "DEF", "MID", "FWD")}
        pick = pd.concat([by["GKP"].head(1), by["DEF"].head(4),
                          by["MID"].head(4), by["FWD"].head(2)])
        assert len(pick) == 11, f"template XI has {len(pick)} players at GW{gw}"
        pts = float(pick["total_points"].sum())
        captain = pick.loc[pick["selected"].idxmax()]
        pts += float(captain["total_points"])
        weekly.append(pts)
    return {"total": round(sum(weekly)), "weekly": [round(w) for w in weekly]}


def random_arm(season: str, max_gw: int = GWS, trials: int = 200) -> dict:
    """Null model: legal random GW1 squad under budget, held, random legal XI."""
    rng = random.Random(SEED)
    snap = replay.snapshot_at(season, 1)
    snap = snap[(snap["price"] > 0) & snap["available"]]
    pools = {p: snap[snap["position"] == p].to_dict("records")
             for p in ("GKP", "DEF", "MID", "FWD")}
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    acts = {gw: replay.actual_points(season, gw) for gw in range(1, max_gw + 1)}

    totals = []
    for _ in range(trials):
        squad = None
        for _ in range(300):                       # rejection-sample a legal squad
            cand, clubs, cost = [], {}, 0.0
            ok = True
            for pos, n in need.items():
                picks = rng.sample(pools[pos], n) if len(pools[pos]) >= n else None
                if picks is None:
                    ok = False; break
                cand += picks
            if not ok:
                continue
            for c in cand:
                clubs[c["team"]] = clubs.get(c["team"], 0) + 1
                cost += float(c["price"])
            if cost <= config.BUDGET and max(clubs.values()) <= config.MAX_PER_CLUB:
                squad = cand; break
        if squad is None:
            continue
        tot = 0.0
        for gw in range(1, max_gw + 1):
            act = acts[gw]
            rows = [(int(c["id"]), str(c["position"]),
                     float(act.get(int(c["id"]), 0))) for c in squad]
            by = {p: [r for r in rows if r[1] == p] for p in need}
            ndef = rng.choice([3, 4, 5]); nmid = rng.choice([3, 4, 5])
            nfwd = 11 - 1 - ndef - nmid
            if not 1 <= nfwd <= 3:
                nfwd = max(1, min(3, nfwd)); nmid = 11 - 1 - ndef - nfwd
            try:
                xi = (rng.sample(by["GKP"], 1) + rng.sample(by["DEF"], ndef)
                      + rng.sample(by["MID"], nmid) + rng.sample(by["FWD"], nfwd))
            except ValueError:
                continue
            tot += sum(r[2] for r in xi) + rng.choice(xi)[2]
        totals.append(tot)
    totals.sort()
    n = len(totals)
    return {"n": n, "mean": round(sum(totals) / n) if n else 0,
            "p50": round(totals[n // 2]) if n else 0,
            "p90": round(totals[int(n * 0.9)]) if n else 0}


# ----------------------------------------------------------------- report
def main() -> None:
    seasons = sys.argv[1:] or ["2023-24", "2024-25"]
    log("# GW1-10 no-leak backtest\n")
    log(f"Seed {SEED}. Pipeline arm only — no LLM in the loop (see report for why).\n")

    log("## Phase 0 — leak audit\n")
    for season in seasons:
        log(f"### {season}")
        have = data_completeness(season)
        missing = [k for k, v in have.items() if not v]
        log(f"- data files: {'all present' if not missing else 'MISSING ' + ', '.join(missing)}")
        prov = strength_provenance(season)
        if prov.get("ok"):
            log("- team strengths: preseason snapshot (played/points/position all zero) — clean")
        else:
            log(f"- team strengths: **NOT preseason** — played={prov.get('played')} "
                f"points={prov.get('points')} position_populated={prov.get('position_populated')}")
        audit = leak_audit(season)
        bad = [r for r in audit if not r["clean"]]
        log(f"- truncation equivalence GW1-{GWS}: "
            f"{'PASS (all identical)' if not bad else 'FAIL at GW ' + ','.join(str(r['gw']) for r in bad)}")
        log("")

    log("## Phase 1 — pipeline arm vs benchmarks\n")
    for season in seasons:
        log(f"### {season}")
        agent = run_arm(season, transfers=True)
        hold = run_arm(season, transfers=False)
        tmpl = template_arm(season)
        rnd = random_arm(season)
        log(f"| arm | GW1-10 total | detail |")
        log(f"|---|---|---|")
        log(f"| pipeline (transfers) | **{agent['total']}** | {agent['moves']} moves, "
            f"{agent['hits']} hit pts, raw {agent['raw']}, "
            f"autosub-aware {agent['autosub_total']} |")
        log(f"| pipeline (hold GW1) | {hold['total']} | no transfers, "
            f"autosub-aware {hold['autosub_total']} |")
        log(f"| template (most-owned XI) | {tmpl['total']} | crowd benchmark |")
        log(f"| random legal squad | {rnd['mean']} | null: p50 {rnd['p50']}, p90 {rnd['p90']}, n={rnd['n']} |")
        log(f"| ceiling (perfect XI from held squad) | {agent['ceiling']} | selection upper bound |")
        log("")
        log(f"- per-GW (pipeline): {agent['weekly']}")
        log(f"- per-GW (template): {tmpl['weekly']}")
        log("")

    # Deliberately NOT agent-backtest-report.md: that file carries hand-written
    # sections (contamination probe, Phase 2, CIs) this script doesn't produce,
    # and overwriting it would silently destroy the evidence base the routine's
    # prompts cite. Merge updated numbers into the report by hand.
    dest = Path(__file__).with_name("agent-backtest-phase01.md")
    dest.write_text("\n".join(OUT) + "\n")
    print(f"\nwrote {dest.name} (merge into agent-backtest-report.md by hand)")


if __name__ == "__main__":
    main()
