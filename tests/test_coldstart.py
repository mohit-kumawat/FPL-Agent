"""Cold-start safety: scoring-regime downweight, uncertainty flags, triggers."""
from __future__ import annotations

import pandas as pd

from fpl_agent import config, daily, data, features, memoryio, models, policy


def _prior_frame(n: int = 12) -> pd.DataFrame:
    """One position, constant process stats, PPG spread — so the ridge collapses
    to a single per-position estimate and the PPG weight is directly observable."""
    rows = []
    for i in range(n):
        rows.append({
            "id": i + 1, "web_name": f"P{i+1}", "element_type": 3,
            "price": 6.0, "minutes": 2500, "starts": 30,
            "points_per_game": 2.0 + i * 0.4,
            "expected_goal_involvements": 5.0, "expected_goals_conceded": 40.0,
            "ict_index": 100.0,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- regime helper
def test_same_generation_is_not_cross_regime():
    assert not config.rules_cross_regime("2023-24", "2024-25")


def test_regime_change_is_detected():
    assert config.rules_cross_regime("2025-26", "2026-27")
    assert config.rules_cross_regime("2024-25", "2025-26")


def test_unknown_season_downweights_conservatively():
    assert config.rules_cross_regime("2030-31", "2031-32")


# ------------------------------------------------------- prior PPG downweight
def test_cross_regime_prior_shrinks_toward_process_stats():
    """With identical process stats, the ridge predicts one value for everyone;
    cross-regime weighting must pull outlier PPGs harder toward it."""
    base = _prior_frame()
    same = base.assign(prior_rules_cross=False)
    cross = base.assign(prior_rules_cross=True)

    p_same = models.prior_baseline(same)
    p_cross = models.prior_baseline(cross)

    hi = base["points_per_game"].idxmax()
    lo = base["points_per_game"].idxmin()
    assert p_cross[hi] < p_same[hi]      # hot outcome trusted less
    assert p_cross[lo] > p_same[lo]      # cold outcome trusted less
    # the explicit argument overrides the frame either way
    assert p_cross.equals(models.prior_baseline(same, cross=True))
    assert p_same.equals(models.prior_baseline(cross, cross=False))


def test_regime_falls_back_to_config_when_frame_is_unmarked(monkeypatch):
    """A frame built outside enrich_players/snapshot_at must not silently get
    same-regime weighting — the season config decides instead."""
    bare = _prior_frame()
    assert "prior_rules_cross" not in bare.columns

    monkeypatch.setattr(config, "PRIOR_SEASON", "2025-26")
    monkeypatch.setattr(config, "CURRENT_SEASON", "2026-27")     # rules changed
    assert models.cross_regime(bare) is True
    assert models.prior_baseline(bare).equals(
        models.prior_baseline(bare.assign(prior_rules_cross=True)))

    monkeypatch.setattr(config, "PRIOR_SEASON", "2023-24")
    monkeypatch.setattr(config, "CURRENT_SEASON", "2024-25")     # same generation
    assert models.cross_regime(bare) is False


def test_marked_frame_wins_over_config(monkeypatch):
    """replay.snapshot_at marks historical frames explicitly; that must beat the
    live-season config, or replaying an old season would use today's regime."""
    monkeypatch.setattr(config, "PRIOR_SEASON", "2025-26")
    monkeypatch.setattr(config, "CURRENT_SEASON", "2026-27")
    marked = _prior_frame().assign(prior_rules_cross=False)
    assert models.cross_regime(marked) is False


def test_enrich_marks_prior_data_only_preseason(monkeypatch):
    teams = [{"id": i, "short_name": f"T{i}", "name": f"Team {i}",
              "strength_attack_home": 1200, "strength_attack_away": 1150,
              "strength_defence_home": 1200, "strength_defence_away": 1150}
             for i in range(1, 3)]
    players = pd.DataFrame([{
        "id": 1, "web_name": "P1", "element_type": 3, "team": 1,
        "price": 6.0, "penalties_order": None,
        "corners_and_indirect_freekicks_order": None,
    }])
    fixtures = [{"event": 1, "team_h": 1, "team_a": 2,
                 "team_h_difficulty": 3, "team_a_difficulty": 3}]
    monkeypatch.setattr(config, "PRIOR_SEASON", "2025-26")
    monkeypatch.setattr(config, "CURRENT_SEASON", "2026-27")

    preseason = {"teams": teams, "events": [
        {"id": 1, "is_next": True, "is_current": False, "finished": False}]}
    df = features.enrich_players(players, preseason, fixtures)
    assert bool(df["prior_rules_cross"].all())

    live = {"teams": teams, "events": [
        {"id": 1, "is_next": False, "is_current": True, "finished": True},
        {"id": 2, "is_next": True, "is_current": False, "finished": False}]}
    df = features.enrich_players(players, live, fixtures)
    assert not df["prior_rules_cross"].any()


# --------------------------------------------------------- uncertainty flags
def _squad_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "web_name": "Premium", "price": 9.0, "minutes": 200, "xmins": 0.9},
        {"id": 2, "web_name": "NewSigning", "price": 5.5, "minutes": 0, "xmins": 0.8},
        {"id": 3, "web_name": "Nailed", "price": 6.0, "minutes": 3000, "xmins": 0.95},
    ])


def test_cheap_new_signing_without_signal_is_flagged():
    flags = policy.uncertainty_flags(_squad_frame(), [1, 2, 3], set(), gws_played=0)
    assert any("NewSigning" in f and "price prior" in f for f in flags)
    assert any("Premium" in f for f in flags)
    assert not any("Nailed" in f for f in flags)


def test_history_flags_do_not_fire_once_minutes_reset_after_gw1():
    """Regression: bootstrap `minutes` holds prior-season totals only until GW1
    is processed, then resets to the current season — so running the history
    checks at gws_played 1-2 flagged every established player as a new signing."""
    established = pd.DataFrame([
        {"id": 1, "web_name": "EstablishedMid", "price": 6.0, "minutes": 90, "xmins": 0.95},
        {"id": 2, "web_name": "EstablishedDef", "price": 5.0, "minutes": 85, "xmins": 0.95},
        {"id": 3, "web_name": "Premium", "price": 12.0, "minutes": 90, "xmins": 0.95},
    ])
    for gws in (1, 2):
        flags = policy.uncertainty_flags(established, [1, 2, 3], set(), gws_played=gws)
        assert not any("new signing" in f or "<1 season of data" in f for f in flags), gws
    # preseason, where minutes IS prior-season data, they still fire
    preseason = policy.uncertainty_flags(established, [1, 2, 3], set(), gws_played=0)
    assert any("new signing" in f for f in preseason)


def test_signal_suppresses_the_cold_start_flag():
    flags = policy.uncertainty_flags(_squad_frame(), [1, 2, 3], {1, 2}, gws_played=0)
    assert not any("NewSigning" in f for f in flags)
    assert not any("Premium" in f for f in flags)


def test_regime_note_appears_during_cold_start_only():
    ep = _squad_frame().assign(prior_rules_cross=True)
    early = policy.uncertainty_flags(ep, [3], set(), gws_played=0)
    late = policy.uncertainty_flags(ep, [3], set(), gws_played=5)
    assert any("scoring rules changed" in f for f in early)
    assert late == []


# -------------------------------------------------- preseason strength guard
def test_zero_strengths_fall_back_to_neutral_not_the_clip_floor():
    """Preseason FPL publishes all-zero strengths; 0/0 is a numpy NaN, not a
    ZeroDivisionError, and used to clamp silently to 0.7 ('hardest fixture')."""
    teams = pd.DataFrame([
        {"id": 1, "strength_attack_home": 0, "strength_attack_away": 0,
         "strength_defence_home": 0, "strength_defence_away": 0},
        {"id": 2, "strength_attack_home": 0, "strength_attack_away": 0,
         "strength_defence_home": 0, "strength_defence_away": 0},
    ]).set_index("id")
    assert features.strength_channels(teams, 1, 2, is_home=True) == (1.0, 1.0)
    assert features.strength_channels(teams, 1, 2, is_home=False) == (1.0, 1.0)


# ------------------------------------------------------------ trigger matrix
def _boot_far_deadline() -> dict:
    return {"events": [{"id": 1, "is_next": True, "is_current": False,
                        "finished": False,
                        "deadline_time": "2099-08-21T17:30:00Z"}]}


def _changes(price_changes: list[dict]) -> dict:
    return {"first_run": False, "new_gw_finished": False, "status_changes": [],
            "news_changes": [], "price_changes": price_changes,
            "gw_state": {"gws_finished": 0}}


def test_owned_player_price_change_triggers_rerun():
    squad = {"players": [{"id": 42}]}
    ch = _changes([{"id": 42, "player": "Mine (T1)", "from": 5.0, "to": 5.1}])
    work = daily.decide_work(ch, _boot_far_deadline(), squad, has_signals=False)
    assert work["models"] and work["optimizer"]
    assert any("selling prices moved" in t for t in work["triggers"])


def test_non_squad_price_change_stays_cheap():
    squad = {"players": [{"id": 42}]}
    ch = _changes([{"id": 7, "player": "Other (T2)", "from": 5.0, "to": 5.1}])
    work = daily.decide_work(ch, _boot_far_deadline(), squad, has_signals=False)
    assert not work["models"]
    assert any("no model rerun" in t for t in work["triggers"])


def test_non_squad_news_no_longer_forces_a_retrain():
    """Regression: the old string-contains matcher was truthy for any labelled
    item, so every news day retrained the models."""
    squad = {"players": [{"id": 42}]}
    ch = _changes([])
    ch["news_changes"] = [{"id": 7, "player": "Other (T2)", "news": "knock"}]
    work = daily.decide_work(ch, _boot_far_deadline(), squad, has_signals=False)
    assert not work["models"]


def test_news_on_a_recommended_target_triggers_a_rerun():
    """Regression: matching owned ids only meant an injury to the player the
    standing advice says to BUY left that advice untouched until the deadline
    window opened."""
    squad = {"players": [{"id": 42}]}
    ch = _changes([])
    ch["status_changes"] = [{"id": 7, "player": "Target (T2)", "from": "a",
                             "to": "i", "news": "out for a month"}]
    work = daily.decide_work(ch, _boot_far_deadline(), squad,
                             has_signals=False, target_ids=[7])
    assert work["models"] and work["optimizer"]
    assert any("recommended transfer target" in t for t in work["triggers"])


def test_owned_player_news_is_labelled_as_squad_news():
    squad = {"players": [{"id": 42}]}
    ch = _changes([])
    ch["status_changes"] = [{"id": 42, "player": "Mine (T1)", "from": "a",
                             "to": "d", "news": "knock"}]
    work = daily.decide_work(ch, _boot_far_deadline(), squad,
                             has_signals=False, target_ids=[7])
    assert work["models"]
    assert any("news on squad" in t for t in work["triggers"])


def test_targets_are_remembered_across_runs(tmp_path, monkeypatch):
    """The trigger above only works if a run records what it recommended."""
    import json
    monkeypatch.setattr(memoryio, "STATE_FILE", tmp_path / "state.json")
    memoryio.save_state({"target_ids": [7, 9]})
    assert json.loads((tmp_path / "state.json").read_text())["target_ids"] == [7, 9]
    assert memoryio.load_state().get("target_ids") == [7, 9]


def test_incoming_transfers_become_the_watched_targets():
    rec = {"transfers": {"plan": {"in": pd.DataFrame([{"id": 7}, {"id": 9}])}}}
    assert daily.recommended_target_ids(rec) == [7, 9]


def test_a_hold_clears_the_targets_rather_than_keeping_stale_ones():
    rec = {"transfers": {"action": "hold", "plan": None}}
    assert daily.recommended_target_ids(rec) == []


def test_the_preseason_build_watches_the_whole_proposed_squad():
    rec = {"initial_build": {"squad": pd.DataFrame([{"id": 1}, {"id": 2}])}}
    assert daily.recommended_target_ids(rec) == [1, 2]


def test_detect_changes_carries_element_ids():
    old = {"events": [], "teams": [{"id": 1, "short_name": "T1"}],
           "elements": [{"id": 9, "web_name": "P", "team": 1, "now_cost": 50,
                         "status": "a", "news": ""}]}
    new = {"events": [], "teams": [{"id": 1, "short_name": "T1"}],
           "elements": [{"id": 9, "web_name": "P", "team": 1, "now_cost": 51,
                         "status": "d", "news": "knock"}]}
    ch = data.detect_changes(new, old)
    assert ch["price_changes"][0]["id"] == 9
    assert ch["status_changes"][0]["id"] == 9
    assert ch["news_changes"][0]["id"] == 9
