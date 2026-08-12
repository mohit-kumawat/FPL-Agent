"""The 2026/27 rulebook, as published by the API and the official announcements.

Verified 2026-08-12 against bootstrap-static `game_config.scoring` / `chips` and
premierleague.com. These tests exist so a rule the code assumes can never drift
silently: the scoring table is checked against the live payload, and the chip
windows are read from it rather than hardcoded.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from fpl_agent import data, memoryio, models, policy, rules, simulate

# The live 2026/27 table, copied from game_config.scoring.
LIVE_SCORING = {
    "long_play": 2, "short_play": 1, "assists": 3, "saves": 1, "bonus": 1,
    "penalties_saved": 5, "penalties_missed": -2, "yellow_cards": -1,
    "red_cards": -3, "own_goals": -2,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
}
# Two full sets of chips, as shipped for 2026/27.
LIVE_CHIPS = [
    {"name": "wildcard", "start_event": 2, "stop_event": 19},
    {"name": "wildcard", "start_event": 20, "stop_event": 38},
    {"name": "freehit", "start_event": 2, "stop_event": 19},
    {"name": "freehit", "start_event": 20, "stop_event": 38},
    {"name": "bboost", "start_event": 1, "stop_event": 19},
    {"name": "bboost", "start_event": 20, "stop_event": 38},
    {"name": "3xc", "start_event": 1, "stop_event": 19},
    {"name": "3xc", "start_event": 20, "stop_event": 38},
]
BOOT = {"game_config": {"scoring": LIVE_SCORING}, "chips": LIVE_CHIPS,
        "events": [{"id": 10, "is_next": True, "is_current": False, "finished": False}]}


# ------------------------------------------------------------------- scoring
def test_model_scoring_matches_the_published_table():
    assert rules.check_scoring(BOOT, models.SCORING_EXPECTED) == []


def test_goalkeepers_earn_no_defensive_contribution_points():
    """game_config.scoring.defensive_contribution.GKP is 0 — awarding keepers 2
    inflated every keeper's EP."""
    assert models.DC_POINTS[1] == 0.0
    assert models.DC_POINTS[2] == models.DC_POINTS[3] == models.DC_POINTS[4] == 2.0


def test_drift_in_any_scoring_value_is_reported():
    expected = dict(models.SCORING_EXPECTED)
    expected["defensive_contribution"] = dict(models.DC_POINTS) | {1: 2.0}
    expected["goals_scored"] = dict(models.GOAL_PTS) | {1: 6}
    found = " ".join(rules.check_scoring(BOOT, expected))
    assert "defensive-contribution points for GKP" in found
    assert "goal points for GKP" in found


def test_goals_conceded_drift_is_reported():
    """Regression: goals_conceded was omitted from the comparison entirely, so
    midfielders starting to lose points for concessions passed unnoticed."""
    sc = dict(LIVE_SCORING)
    sc["goals_conceded"] = {"GKP": -1, "DEF": -1, "MID": -1, "FWD": 0}
    found = " ".join(rules.check_scoring({**BOOT, "game_config": {"scoring": sc}},
                                         models.SCORING_EXPECTED))
    assert "goals-conceded points for MID" in found


def test_scalar_drift_on_the_MODEL_side_is_reported():
    """Regression: scalars were compared against literals inside the checker, so
    editing the model's value left the tripwire reporting no drift."""
    expected = dict(models.SCORING_EXPECTED) | {"assists": 4.0}
    found = " ".join(rules.check_scoring(BOOT, expected))
    assert "assist points" in found and "model uses 4" in found


def test_unverifiable_rules_are_named_not_claimed():
    """The API publishes per-unit values but not the divisors the model applies,
    so those must be declared unverifiable rather than reported as checked."""
    joined = " ".join(rules.UNVERIFIABLE_RULES)
    assert "saves per point" in joined and "concessions per point" in joined


def test_missing_game_config_is_silent_not_noisy():
    assert rules.check_scoring({}, models.SCORING_EXPECTED) == []


def test_keeper_dc_column_cannot_add_points():
    """Even if the data starts populating keeper defensive actions, the points
    must stay zero."""
    df = pd.DataFrame([
        {"id": 1, "element_type": 1, "minutes": 900, "defensive_contribution": 500,
         "expected_goals": 0.0, "expected_assists": 0.0,
         "expected_goals_conceded": 9.0, "bonus": 0, "saves": 0},
        {"id": 2, "element_type": 2, "minutes": 900, "defensive_contribution": 500,
         "expected_goals": 0.0, "expected_assists": 0.0,
         "expected_goals_conceded": 9.0, "bonus": 0, "saves": 0},
    ])
    comp = models.component_frame(df)
    gk_neutral = comp.loc[0, "neutral90"]
    df_neutral = comp.loc[1, "neutral90"]
    # the defender banks DC points, the keeper does not
    assert df_neutral > gk_neutral + 1.0


# --------------------------------------------------------------------- chips
def test_chip_windows_come_from_the_api():
    w = rules.chip_windows(BOOT)
    assert w["wildcard"] == [(2, 19), (20, 38)]
    assert w["bboost"] == [(1, 19), (20, 38)]
    assert w["3xc"] == [(1, 19), (20, 38)]
    assert w["freehit"] == [(2, 19), (20, 38)]


def test_bench_boost_and_triple_captain_are_playable_in_gw1():
    """Regression: the code claimed no chip was playable in GW1, which is only
    true of Wildcard and Free Hit."""
    w = rules.chip_windows(BOOT)
    full = ["wildcard1", "freehit1", "bboost1", "3xc1"]
    assert sorted(rules.playable_now(full, 1, w)) == ["3xc", "bboost"]


def test_second_half_copies_are_not_playable_in_the_first_half():
    w = rules.chip_windows(BOOT)
    assert rules.playable_now(["bboost2", "3xc2"], 5, w) == []
    assert sorted(rules.playable_now(["bboost2", "3xc2"], 25, w)) == ["3xc", "bboost"]


def test_used_first_half_chip_leaves_the_second_half_copy():
    w = rules.chip_windows(BOOT)
    remaining = ["wildcard2", "freehit2", "bboost2", "3xc2"]
    assert rules.playable_now(remaining, 10, w) == []
    assert len(rules.playable_now(remaining, 22, w)) == 4


def test_unsuffixed_chip_names_still_work():
    """Back-compat with squad files written before the two-set era."""
    w = rules.chip_windows(BOOT)
    assert sorted(rules.playable_now(["bboost", "3xc"], 10, w)) == ["3xc", "bboost"]


def test_valid_chip_names_cover_both_sets():
    names = rules.valid_chip_names()
    for family in rules.CHIP_FAMILIES:
        assert family in names and f"{family}1" in names and f"{family}2" in names


def _xi(cap_ep: float, cap_fx: int, bench_ep: float) -> dict:
    return {"captain": pd.Series({"web_name": "C", "ep_next": cap_ep,
                                  "next_n_fixtures": cap_fx, "id": 1}),
            "xi": pd.DataFrame([{"id": 1, "web_name": "C", "ep_next": cap_ep}]),
            "vice": pd.Series({"web_name": "V", "id": 2}),
            "bench_order": pd.DataFrame([{"id": 3, "ep_next": bench_ep,
                                          "next_n_fixtures": cap_fx}])}


def _boot_at(gw: int) -> dict:
    return {**BOOT, "events": [{"id": gw, "is_next": True, "is_current": False,
                                "finished": False}]}


def test_only_one_chip_is_recommended_per_gameweek():
    """FPL allows one chip per gameweek; two NOW recommendations is illegal advice."""
    notes = policy.chip_advice(_boot_at(10), _xi(14.0, 2, 20.0),
                               ["3xc1", "bboost1"], scenarios=[
                                   {"gw": 30, "kind": "double", "prob": 0.2}])
    nows = [n for n in notes if "NOW" in n]
    assert len(nows) == 1
    assert any("only one chip per gameweek" in n for n in notes)
    assert any("blocked this GW" in n for n in notes)


def test_a_first_half_chip_cannot_be_played_after_the_split():
    """Unused first-half chips expire at the split — they do not carry over."""
    notes = policy.chip_advice(_boot_at(24), _xi(14.0, 2, 20.0), ["3xc1", "bboost1"])
    assert len(notes) == 1 and "No chip is playable in GW24" in notes[0]


def test_gw1_advice_covers_the_playable_chips_only():
    notes = policy.chip_advice(_boot_at(1), _xi(8.0, 1, 6.0),
                               ["wildcard1", "freehit1", "bboost1", "3xc1"])
    text = " ".join(notes)
    assert "Triple Captain" in text and "Bench Boost" in text
    assert "Wildcard" not in text and "Free Hit" not in text


def test_no_playable_chip_says_so_without_inventing_advice():
    notes = policy.chip_advice(_boot_at(10), _xi(8.0, 1, 6.0), ["bboost2", "3xc2"])
    assert len(notes) == 1 and "No chip is playable in GW10" in notes[0]


def test_zero_ep_never_reads_as_play():
    """A blank gameweek (or missing fixture data) makes both EVs zero, where
    `now >= hold` is trivially true and used to print 'play'."""
    notes = policy.chip_advice(_boot_at(28), _xi(0.0, 1, 0.0), ["3xc1", "bboost1"])
    assert not any("NOW" in n for n in notes)
    assert all("hold" in n for n in notes if "Triple Captain" in n or "Bench Boost" in n)


# ------------------------------------------------------------------ lockdown
def _ev(**kw) -> dict:
    return {"id": 5, "deadline_time": "2026-09-19T11:00:00Z", **kw}


FX = [{"event": 5, "kickoff_time": "2026-09-19T14:00:00Z"},
      {"event": 5, "kickoff_time": "2026-09-21T19:00:00Z"}]   # Monday night


def test_points_are_not_final_before_the_0900_uk_lockdown():
    """2026/27 moved lockdown to 09:00 UK the day after the last match, so late
    bonus and DC corrections land after `finished` flips."""
    final, why = data.gw_points_final(
        _ev(finished=True), FX, now=datetime(2026, 9, 21, 22, 0, tzinfo=timezone.utc))
    assert not final and "lockdown" in why


def test_points_are_final_after_the_lockdown():
    final, _ = data.gw_points_final(
        _ev(finished=True), FX, now=datetime(2026, 9, 22, 9, 30, tzinfo=timezone.utc))
    assert final


def test_lockdown_is_measured_from_the_last_match_not_the_deadline():
    """A Monday-night finish locks down Tuesday, not Sunday — measuring from the
    deadline would call the gameweek final two days early."""
    day_after_deadline = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    assert not data.gw_points_final(_ev(finished=True), FX, now=day_after_deadline)[0]


def test_data_checked_flag_wins_immediately():
    final, why = data.gw_points_final(
        _ev(finished=True, data_checked=True), FX,
        now=datetime(2026, 9, 21, 22, 0, tzinfo=timezone.utc))
    assert final and "data_checked" in why


def test_missing_fixtures_declines_rather_than_guessing_early():
    final, why = data.gw_points_final(_ev(finished=True), [],
                                      now=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert not final and "data_checked" in why


def test_unfinished_gameweek_is_never_final():
    assert not data.gw_points_final(_ev(finished=False), FX)[0]


# ------------------------------------------------------------- simulator sync
def test_simulator_gives_keepers_no_dc_points():
    from test_pipeline import _players
    df = _players()
    df["defensive_contribution"] = 400.0
    ep = models.expected_points(df)
    gk = ep[ep["element_type"] == 1]
    sim = simulate.simulate_players(gk, n_trials=800, seed=5)
    # keepers score from saves/CS/appearance only; a 400-DC keeper must not
    # out-earn his own ep_next by the 2-point DC award
    assert (sim["sim_mean"] <= gk["ep_next"] + 1.5).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------- calibration retry (fix 1)
_CAL_FX = [{"event": 5, "kickoff_time": "2026-09-20T15:30:00Z"}]   # Sunday 16:30
_CAL_BOOT = {"events": [
    {"id": 5, "finished": True, "deadline_time": "2026-09-19T11:00:00Z"},
    {"id": 6, "is_next": True, "is_current": False, "finished": False,
     "deadline_time": "2099-09-26T11:00:00Z"}]}


_SUNDAY_EVE = datetime(2026, 9, 20, 20, 0, tzinfo=timezone.utc)
_MONDAY_AM = datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc)


def test_pending_calibration_is_empty_before_the_lockdown():
    """Sunday evening: the gameweek is finished but points are not final yet."""
    from fpl_agent import daily as daily_mod
    assert daily_mod.pending_calibration(_CAL_BOOT, _CAL_FX, {},
                                         now=_SUNDAY_EVE) == []


def test_pending_calibration_finds_the_gameweek_once_points_are_final():
    """Monday: new_gw_finished is False, so this is the ONLY thing that can
    re-trigger the calibration that Sunday's run deferred."""
    from fpl_agent import daily as daily_mod
    assert daily_mod.pending_calibration(_CAL_BOOT, _CAL_FX, {},
                                         now=_MONDAY_AM) == [5]


def test_a_scored_gameweek_is_not_calibrated_twice():
    from fpl_agent import daily as daily_mod
    assert daily_mod.pending_calibration(
        _CAL_BOOT, _CAL_FX, {"scored_gws": [5]}, now=_MONDAY_AM) == []


def test_pending_calibration_triggers_a_retrain():
    """Regression: without this trigger a deferred gameweek was never scored,
    because full_retrain only ever fired on the day the gameweek finished."""
    from fpl_agent import daily as daily_mod
    changes = {"first_run": False, "new_gw_finished": False,
               "status_changes": [], "news_changes": [], "price_changes": [],
               "gw_state": {"gws_finished": 5}}
    boot = {"events": [{"id": 6, "is_next": True, "is_current": False,
                        "finished": False, "deadline_time": "2099-09-26T11:00:00Z"}]}
    work = daily_mod.decide_work(changes, boot, {"players": []},
                                 has_signals=False, pending_gws=[5])
    assert work["models"] and work["full_retrain"]
    assert any("points now final" in t for t in work["triggers"])


# ------------------------------------------------- the agent's own contract
def test_example_squad_chip_names_are_all_valid():
    """squad.example.yaml is what an owner copies; every chip name in it must
    pass the validator, or a fresh install starts with a verify warning."""
    import pathlib

    import yaml
    doc = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "squad.example.yaml").read_text())
    chips = {str(c).strip().lower() for c in (doc.get("chips_available") or [])}
    assert chips, "example squad lists no chips"
    assert chips <= rules.valid_chip_names(), chips - rules.valid_chip_names()
    # both halves of every family should be present in a fresh season
    for family in rules.CHIP_FAMILIES:
        assert f"{family}1" in chips and f"{family}2" in chips, family


def test_agent_runbook_documents_the_rules_it_must_operate_under():
    """AGENT.md is the agent's system prompt. If these facts fall out of it the
    agent silently loses them, so the important ones are pinned here."""
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parent.parent / "AGENT.md").read_text()
    for needle in (
        "09:00 UK",                      # lockdown timing
        "calibration deferred",          # expected, not a fault
        "BPS was reworked",              # stale prior-season bonus patterns
        "SCORING RULE DRIFT",            # stop-and-escalate signal
        "2 January",                     # first-half chip expiry
        "No extra December transfers",   # no AFCON allocation
        "saves-per-point",               # named as un-auto-checkable
    ):
        assert needle in doc, f"AGENT.md no longer documents: {needle}"
    # the role vocabulary the agent is told to prefer must actually exist
    for role in memoryio.SIGNAL_ROLES:
        assert role in doc, f"AGENT.md omits signal role: {role}"
