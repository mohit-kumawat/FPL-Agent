"""Score a picked team under FPL's real matchday rules.

The strategy replays previously scored a gameweek as "XI actuals + doubled
captain", ignoring automatic substitutions and the vice-captain fallback. Both
exist in the real game, so replay totals understated what the same picks would
actually have banked. This module applies the real rules:

  - a starter with 0 minutes is replaced from the bench, in bench order,
    provided the resulting formation stays legal (1 GKP / >=3 DEF / >=2 MID /
    >=1 FWD); goalkeepers swap only with goalkeepers
  - the captain's double falls to the vice when the captain plays 0 minutes;
    if both blank, nobody doubles

Consumers pass per-player minutes for the gameweek (a double GW sums both
matches), so "played" means "played at some point in the GW" — the same rule
FPL applies.
"""
from __future__ import annotations

import pandas as pd

XI_MINIMA = {1: 1, 2: 3, 3: 2, 4: 1}   # GKP, DEF, MID, FWD floor in a legal XI


def gw_score(xi_result: dict, minutes: pd.Series, act: pd.Series) -> dict:
    """Score one GW for a picked team.

    xi_result: optimizer.pick_xi output (xi, bench_order, captain, vice).
    minutes:   GW minutes per element id (missing id = did not play).
    act:       GW actual points per element id.

    Returns {"raw", "autosub", "subs", "vice_used"} — `raw` is the legacy
    no-autosub score kept for comparability with previously published numbers.
    """
    xi = xi_result["xi"]
    bench = xi_result["bench_order"]
    cap_id = int(xi_result["captain"]["id"])
    vice_id = int(xi_result["vice"]["id"])

    def played(pid: int) -> bool:
        return float(minutes.get(pid, 0)) > 0

    def pts(pid: int) -> float:
        return float(act.get(pid, 0))

    # ---- legacy raw score (XI + doubled captain, no matchday rules) --------
    raw = float(xi["id"].map(act).fillna(0).sum()) + pts(cap_id)

    # ---- automatic substitutions ------------------------------------------
    final = {int(r.id): int(r.element_type) for r in xi.itertuples() if played(int(r.id))}
    missing = {int(r.id): int(r.element_type) for r in xi.itertuples() if not played(int(r.id))}
    counts = {et: 0 for et in XI_MINIMA}
    for et in list(final.values()) + list(missing.values()):
        counts[et] += 1                      # counts of the ORIGINAL XI

    subs = 0
    for b in bench.itertuples():             # bench priority order
        if not missing:
            break
        bid, bet = int(b.id), int(b.element_type)
        if not played(bid):
            continue
        # goalkeepers swap only with the goalkeeper slot
        candidates = [m for m, met in missing.items()
                      if (met == 1) == (bet == 1)]
        for m in candidates:
            met = missing[m]
            counts[met] -= 1
            counts[bet] += 1
            if all(counts[et] >= XI_MINIMA[et] for et in XI_MINIMA):
                del missing[m]
                final[bid] = bet
                subs += 1
                break
            counts[met] += 1                 # illegal — undo and try the next
            counts[bet] -= 1

    autosub = float(sum(pts(pid) for pid in final))
    # remaining missing starters score their (zero-minute) points as-is
    autosub += float(sum(pts(pid) for pid in missing))

    # ---- armband -----------------------------------------------------------
    vice_used = False
    if played(cap_id):
        autosub += pts(cap_id)
    elif played(vice_id):
        autosub += pts(vice_id)
        vice_used = True

    return {"raw": round(raw, 1), "autosub": round(autosub, 1),
            "subs": subs, "vice_used": vice_used}
