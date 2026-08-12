"""Phase 2 — emit captain-choice candidate sets for the blind-label test.

Two views of the SAME pre-deadline rows for each GW:
  named/  player names visible  -> recall is available to the agent
  anon/   names replaced by opaque codes, teams too -> recall is impossible

The agent picks a captain from each view independently. If the named arm beats
the anonymous arm, the agent was using memory of the season rather than the
features in front of it. That comparison is robust to the agent already knowing
the season, which is why it is the test that survives contamination.

Emits candidates only. Actual points are never touched here.

Usage: uv run python eval/phase2_candidates.py <season> <n_gws>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import replay  # noqa: E402

TOP_N = 8
FEATURES = ["ep_next", "ep_horizon", "xmins", "price", "next_gw_mult",
            "roll_points", "points_per_game", "minutes"]


def code(name: str, salt: str, taken: set[str] | None = None) -> str:
    """Stable opaque label. Salted per GW so codes can't be linked across GWs.

    `taken` guards against two candidates sharing a web_name in one GW (the
    hash would collide and the scorer would silently score the wrong player);
    a collision re-salts until unique.
    """
    n = 0
    while True:
        raw = f"{salt}:{name}" + (f":{n}" if n else "")
        c = "P" + hashlib.sha256(raw.encode()).hexdigest()[:6].upper()
        if taken is None or c not in taken:
            if taken is not None:
                taken.add(c)
            return c
        n += 1


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
    n_gws = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = {"season": season, "gws": {}}

    for gw in range(1, n_gws + 1):
        ep = replay.expected_points_at(season, gw)
        ep = ep[ep["available"] & (ep["xmins"] > 0.4)]
        top = ep.nlargest(TOP_N, "ep_next")

        named, anon = [], []
        taken: set[str] = set()
        for r in top.itertuples():
            feats = {f: (round(float(getattr(r, f)), 3)
                         if hasattr(r, f) and getattr(r, f) == getattr(r, f) else None)
                     for f in FEATURES}
            named.append({"id": int(r.id), "name": str(r.web_name),
                          "team": str(r.team_short), "pos": str(r.position), **feats})
            anon.append({"code": code(str(r.web_name), f"{season}-{gw}", taken),
                         "team": code(str(r.team_short), f"T{season}-{gw}"),
                         "pos": str(r.position), **feats})

        out["gws"][str(gw)] = {
            "named": named,
            "anon": anon,
            # the pipeline's own pick = highest ep_next, for the baseline arm
            "pipeline_pick_id": int(top.iloc[0]["id"]),
            "pipeline_pick_name": str(top.iloc[0]["web_name"]),
        }

    dest = Path(__file__).with_name(f"phase2-candidates-{season}.json")
    dest.write_text(json.dumps(out, indent=1))

    # The anon view goes to its OWN file, order-shuffled per GW. Printing anon
    # and named adjacently (as the first version did) made the mapping
    # recoverable by index, which gutted the "recall impossible" premise --
    # give the anon judge this file and nothing else.
    import random
    rng = random.Random(f"shuffle-{season}")
    lines = [f"# Anonymous captain candidates, {season} GW1-{n_gws}.",
             "# Pick one code per GW from features alone.", ""]
    for gw in range(1, n_gws + 1):
        rows = list(out["gws"][str(gw)]["anon"])
        rng.shuffle(rows)
        lines.append(f"GW{gw}: " + " | ".join(
            f"{c['code']}({c['pos']},{c['price']},ep{c['ep_next']},xm{c['xmins']})"
            for c in rows))
    view = Path(__file__).with_name(f"phase2-anon-{season}.txt")
    view.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest.name} (mapping — keep from the anon judge)")
    print(f"wrote {view.name} (shuffled anon view — the only thing the judge sees)")


if __name__ == "__main__":
    main()
