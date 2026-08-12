"""Matchday-rules scorer: autosubs, formation legality, vice-captain fallback."""
from __future__ import annotations

import pandas as pd

from fpl_agent import scoring


def _team(bench_types: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    """A 3-4-3: GK(1) DEF(2,3,4) MID(5,6,7,8) FWD(9,10,11); bench ids 12-15
    with configurable positions (default GK, DEF, MID, FWD). Captain 5, vice 9."""
    xi = pd.DataFrame([
        {"id": 1, "element_type": 1}, {"id": 2, "element_type": 2},
        {"id": 3, "element_type": 2}, {"id": 4, "element_type": 2},
        {"id": 5, "element_type": 3}, {"id": 6, "element_type": 3},
        {"id": 7, "element_type": 3}, {"id": 8, "element_type": 3},
        {"id": 9, "element_type": 4}, {"id": 10, "element_type": 4},
        {"id": 11, "element_type": 4},
    ])
    bench = pd.DataFrame([{"id": 12 + i, "element_type": et}
                          for i, et in enumerate(bench_types)])
    return {"xi": xi, "bench_order": bench,
            "captain": {"id": 5}, "vice": {"id": 9}}


def _gw(zero: set[int], **overrides: float) -> tuple[pd.Series, pd.Series]:
    """(minutes, points): players in `zero` played 0' and scored 0 (as in the
    real game); everyone else played 90' for 2 pts unless overridden (p<id>=pts)."""
    mins = pd.Series({i: 0 if i in zero else 90 for i in range(1, 16)})
    acts = {i: 0.0 if i in zero else 2.0 for i in range(1, 16)}
    acts.update({int(k[1:]): v for k, v in overrides.items()})
    return mins, pd.Series(acts)


def test_everyone_plays_matches_the_raw_score():
    r = scoring.gw_score(_team(), *_gw(set(), p5=10))
    assert r["raw"] == r["autosub"] == 30 + 10   # XI 30, captain doubles
    assert r["subs"] == 0 and not r["vice_used"]


def test_blank_starter_is_replaced_in_bench_order():
    r = scoring.gw_score(_team(), *_gw({6}, p13=6))
    assert r["subs"] == 1
    assert r["autosub"] == r["raw"] + 6          # DEF 13 comes on for MID 6


def test_formation_floor_blocks_an_illegal_sub():
    # all three FWDs blank; bench is GK, DEF, DEF, MID. Two subs are legal
    # (FWD count 3 -> 1) but the third would take FWD to 0 < 1, so the played
    # bench MID stays off.
    r = scoring.gw_score(_team(bench_types=(1, 2, 2, 3)), *_gw({9, 10, 11}))
    assert r["subs"] == 2
    assert r["autosub"] == r["raw"] + 4          # two bench 2-pointers on


def test_goalkeeper_swaps_only_with_goalkeeper():
    r = scoring.gw_score(_team(), *_gw({1}, p12=7))
    assert r["subs"] == 1
    assert r["autosub"] == r["raw"] + 7


def test_outfielder_never_replaces_the_goalkeeper():
    # GK blanks and the bench GK blanks too: nobody can come on
    r = scoring.gw_score(_team(), *_gw({1, 12}))
    assert r["subs"] == 0
    assert r["autosub"] == r["raw"]


def test_bench_player_who_also_blanked_cannot_come_on():
    r = scoring.gw_score(_team(), *_gw({6, 13, 14, 15}))
    assert r["subs"] == 0
    assert r["autosub"] == r["raw"]


def test_vice_doubles_when_captain_blanks():
    r = scoring.gw_score(_team(), *_gw({5}, p9=8))
    assert r["vice_used"] and r["subs"] == 1
    # raw: XI 9x2 + 8 + blank 0, captain adds nothing = 26
    # autosub: bench DEF on (+2), vice doubles (+8)
    assert r["raw"] == 26
    assert r["autosub"] == 36


def test_captain_and_vice_both_blank_nobody_doubles():
    r = scoring.gw_score(_team(), *_gw({5, 9}, p13=3, p14=4))
    assert not r["vice_used"]
    assert r["subs"] == 2
    assert r["autosub"] == r["raw"] + 3 + 4
