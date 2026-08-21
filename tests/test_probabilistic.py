"""Phase C: minutes distribution, Monte Carlo outcomes, modes, chips, roles."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from fpl_agent import memoryio, models, optimizer, policy, simulate
from test_pipeline import _players


# ------------------------------------------------------ minutes distribution
def test_minutes_distribution_fields_are_coherent():
    ep = models.expected_points(_players())
    for c in ("p_start", "p_60", "p_bench", "expected_minutes", "minutes_sd"):
        assert c in ep.columns and ep[c].notna().all()
    assert ep["p_start"].between(0, 1).all()
    assert ep["p_bench"].between(0, 1).all()
    assert (ep["p_60"] <= (ep["xmins"] / 0.60).clip(0, 1) + 1e-9).all()
    assert (ep["expected_minutes"] - ep["xmins"] * 90).abs().max() < 1e-9


def test_start_and_bench_probability_sum_to_the_appearance_probability():
    """p_start + p_bench is P(appears) = p_play, NOT 1.0. Players under an xmins
    of 0.60 keep genuine zero-minutes mass; only those at or above it are treated
    as certain to appear. Pinning this stops the two readings drifting apart."""
    ep = models.expected_points(_players())
    p_play = (ep["xmins"] / 0.60).clip(0, 1)
    assert ((ep["p_start"] + ep["p_bench"]) - p_play).abs().max() < 1e-9
    assert (ep["p_start"] <= p_play + 1e-9).all()


def test_minutes_signal_moves_p_start_with_xmins():
    """A signal that raises xmins must raise p_start too. Regression: roles set
    only the xmins bound, so p_start kept its stale price-prior value and a
    signalled starter read as 'certain to appear, probably off the bench'."""
    df = _players()
    # a new signing with no history: xmins comes from the price prior, ~0.30
    df.loc[df["id"] == 8, ["minutes", "starts"]] = [0, 0]
    plain = models.expected_points(df)
    base = plain.loc[plain["id"] == 8].iloc[0]

    up = pd.DataFrame({"xmins_min": [0.85]}, index=[8])
    lifted = models.expected_points(df, None, up)
    row = lifted.loc[lifted["id"] == 8].iloc[0]

    assert row["xmins"] > base["xmins"]
    assert row["p_start"] > base["p_start"], "p_start ignored the minutes signal"
    assert row["p_start"] > 0.6, "an expected starter should read as likely to start"
    assert abs((row["p_start"] + row["p_bench"]) - 1.0) < 1e-9


def test_minutes_cap_pulls_p_start_down():
    """The mirror case: a bench_role cap must not leave a player reading as a
    75%-likely starter with zero chance of a substitute appearance."""
    df = _players()
    down = pd.DataFrame({"xmins_max": [0.45]}, index=[8])
    capped = models.expected_points(df, None, down)
    row = capped.loc[capped["id"] == 8].iloc[0]

    assert row["xmins"] == pytest.approx(0.45)
    assert row["p_start"] < 0.5, "a capped player still read as a likely starter"
    assert row["p_bench"] > 0, "a bench role must carry substitute-appearance mass"


def test_p_start_stays_inside_the_band_xmins_allows():
    """p_start must be reachable from xmins: at least the value implied when every
    start is a full 90 and every cameo is SUB_MINUTES_SHARE long, at most p_play."""
    for adjust in (None,
                   pd.DataFrame({"xmins_min": [0.85]}, index=[8]),
                   pd.DataFrame({"xmins_max": [0.45]}, index=[8])):
        ep = models.expected_points(_players(), None, adjust)
        p_play = (ep["xmins"] / 0.60).clip(0, 1)
        floor = ((ep["xmins"] - models.SUB_MINUTES_SHARE * p_play)
                 / (1.0 - models.SUB_MINUTES_SHARE)).clip(lower=0)
        assert (ep["p_start"] >= floor - 1e-9).all()
        assert (ep["p_start"] <= p_play + 1e-9).all()


def test_minutes_sd_peaks_in_the_rotation_zone():
    df = _players()
    nailed = df.copy(); nailed["minutes"] = 3400; nailed["starts"] = 38
    ep = models.expected_points(nailed)
    rotation_sd = ep.loc[ep["xmins"].between(0.45, 0.7), "minutes_sd"]
    nailed_sd = ep.loc[ep["xmins"] > 0.95, "minutes_sd"]
    if len(rotation_sd) and len(nailed_sd):
        assert rotation_sd.min() > nailed_sd.max()


def test_xmins_values_are_unchanged_by_the_distribution_layer():
    """Backward compatibility: the scalar the optimizer consumes must be the
    same number expected_minutes_fraction always produced."""
    df = _players()
    ep = models.expected_points(df)
    legacy = models.expected_minutes_fraction(df.assign(
        roll_minutes=float("nan"), roll_starts=float("nan")))
    assert (ep["xmins"] - legacy).abs().max() < 1e-9


# ---------------------------------------------------------------- simulation
def test_simulation_is_deterministic_under_a_seed():
    ep = models.expected_points(_players())
    a = simulate.simulate_players(ep, n_trials=500, seed=7)
    b = simulate.simulate_players(ep, n_trials=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_simulation_mean_tracks_ep_and_blank_is_zero():
    df = _players()
    df.loc[df["team"] == 1, ["next_gw_mult", "next_att_mult",
                             "next_def_mult", "next_n_fixtures"]] = 0
    ep = models.expected_points(df)
    sim = simulate.simulate_players(ep, n_trials=1500, seed=1)
    blanked = sim[ep["team"] == 1]
    assert (blanked["sim_mean"] == 0).all()
    assert (blanked["p_blank"] == 1.0).all()
    # playing players: simulated mean correlates with the EP model
    playing = ep["team"] != 1
    rho = ep.loc[playing, "ep_next"].rank().corr(sim.loc[playing, "sim_mean"].rank())
    assert rho > 0.5


def test_haul_probability_orders_with_upside():
    ep = models.expected_points(_players())
    sim = simulate.simulate_players(ep, n_trials=1500, seed=2)
    hi = sim.loc[ep["ep_next"].idxmax(), "p_haul"]
    lo = sim.loc[ep["ep_next"].idxmin(), "p_haul"]
    assert hi > lo
    assert (sim["sim_p10"] <= sim["sim_p50"]).all()
    assert (sim["sim_p50"] <= sim["sim_p90"]).all()


# ------------------------------------------------------------- strategy modes
def _xi_with_differential():
    ep = models.expected_points(_players(8))
    # make one XI-quality player a big differential within the tolerance
    ep["selected_by_percent"] = 50.0
    xi = optimizer.pick_xi(optimizer.build_squad(ep)["squad"])
    cap_ep = float(xi["captain"]["ep_next"])
    third = xi["xi"].nlargest(3, "ep_next").iloc[2]
    ep.loc[ep["id"] == third["id"], "selected_by_percent"] = 3.0
    ep.loc[ep["id"] == third["id"], "ep_next"] = cap_ep - 0.4
    xi["xi"].loc[xi["xi"]["id"] == third["id"], "ep_next"] = cap_ep - 0.4
    return ep, xi


def test_safe_mode_never_swaps_to_the_differential():
    ep, xi = _xi_with_differential()
    cap = policy.assess_captain(xi, ep, mode="safe")
    assert cap["pick"] == xi["captain"]["web_name"]
    assert cap.get("safe_pick") is None
    assert cap["differential"] is not None      # surfaced, not picked


def test_chase_mode_takes_the_close_differential_and_shows_the_safe_pick():
    ep, xi = _xi_with_differential()
    cap = policy.assess_captain(xi, ep, mode="chase")
    assert cap["pick"] == cap["differential"]["pick"]
    assert cap["safe_pick"] == xi["captain"]["web_name"]


def test_unknown_mode_falls_back_to_safe():
    ep, xi = _xi_with_differential()
    cap = policy.assess_captain(xi, ep, mode="yolo")
    assert cap["mode"] == "safe"
    assert cap["pick"] == xi["captain"]["web_name"]


def _tiny_xi(diff_is_vice: bool) -> tuple[pd.DataFrame, dict]:
    xi = pd.DataFrame([
        {"id": 1, "web_name": "Cap", "element_type": 3, "ep_next": 6.0, "ep_sd": 1.0},
        {"id": 2, "web_name": "Diff", "element_type": 3, "ep_next": 5.7, "ep_sd": 1.0},
        {"id": 3, "web_name": "Third", "element_type": 4, "ep_next": 4.0, "ep_sd": 1.0},
    ])
    vice = xi.iloc[1] if diff_is_vice else xi.iloc[2]
    players = xi.assign(selected_by_percent=[50.0, 3.0, 40.0])
    return players, {"xi": xi, "captain": xi.iloc[0], "vice": vice,
                     "bench_order": pd.DataFrame([{"id": 9, "ep_next": 1.0}])}


def test_chase_mode_never_makes_captain_and_vice_the_same_player():
    """Regression: chase swapped the armband without moving the vice, so the
    differential could be recommended as both — a team FPL will not accept."""
    players, xi = _tiny_xi(diff_is_vice=True)
    cap = policy.assess_captain(xi, players, mode="chase")
    assert cap["pick"] == "Diff"
    assert cap["vice"] != cap["pick"]
    assert cap["vice"] == "Cap"          # the EP-max pick becomes the fallback


def test_chase_mode_leaves_an_unrelated_vice_alone():
    players, xi = _tiny_xi(diff_is_vice=False)
    cap = policy.assess_captain(xi, players, mode="chase")
    assert cap["pick"] == "Diff" and cap["vice"] == "Third"


def test_duplicate_web_name_does_not_break_the_sim_merge():
    """Regression: keying the simulation by web_name made .loc return a frame for
    duplicated names (two Wards), raising TypeError and aborting the daily run."""
    xi = pd.DataFrame([
        {"id": 1, "web_name": "Ward", "element_type": 3, "ep_next": 6.0, "ep_sd": 1.0},
        {"id": 2, "web_name": "Ward", "element_type": 2, "ep_next": 5.0, "ep_sd": 1.0},
    ])
    xi_result = {"xi": xi, "captain": xi.iloc[0], "vice": xi.iloc[1],
                 "bench_order": pd.DataFrame()}
    players = xi.assign(selected_by_percent=[50.0, 40.0])
    sim = pd.DataFrame([
        {"id": 1, "web_name": "Ward", "sim_p50": 5, "sim_p10": 1, "sim_p90": 12,
         "p_haul": 0.2, "p_blank": 0.1},
        {"id": 2, "web_name": "Ward", "sim_p50": 4, "sim_p10": 1, "sim_p90": 9,
         "p_haul": 0.1, "p_blank": 0.2},
    ])
    cap = policy.assess_captain(xi_result, players, mode="safe", sim=sim)
    # the CAPTAIN's row (id 1), not whichever duplicate label sorted first
    assert cap["simulation"]["Ward"]["median"] == 5.0


# -------------------------------------------------------------------- chips
def _boot(gw: int = 10) -> dict:
    return {"events": [{"id": gw, "is_next": True, "is_current": False,
                        "finished": False}]}


def _xi(cap_ep: float, cap_fx: int, bench_ep: float = 6.0) -> dict:
    return {"captain": pd.Series({"web_name": "Prem", "ep_next": cap_ep,
                                  "next_n_fixtures": cap_fx, "id": 1}),
            "xi": pd.DataFrame([{"id": 1, "web_name": "Prem", "ep_next": cap_ep}]),
            "vice": pd.Series({"web_name": "V", "id": 2}),
            "bench_order": pd.DataFrame([{"id": 3, "ep_next": bench_ep}])}


def test_triple_captain_on_a_double_beats_a_thin_future():
    scen = [{"gw": 29, "kind": "double", "prob": 0.3}]
    notes = policy.chip_advice(_boot(24), _xi(cap_ep=12.0, cap_fx=2),
                               ["3xc"], scenarios=scen)
    assert any("Triple Captain" in n and "NOW" in n for n in notes)


def test_triple_captain_holds_for_a_likely_double():
    scen = [{"gw": 29, "kind": "double", "prob": 0.9}]
    notes = policy.chip_advice(_boot(10), _xi(cap_ep=6.0, cap_fx=1),
                               ["3xc"], scenarios=scen)
    assert any("Triple Captain" in n and "hold" in n for n in notes)
    assert not any("NOW" in n for n in notes)


def test_no_scenarios_assumes_a_double_early_but_not_late():
    early = policy.chip_advice(_boot(10), _xi(6.0, 1), ["3xc"])
    late = policy.chip_advice(_boot(35), _xi(6.0, 1), ["3xc"])
    assert any("assuming a usable double" in n for n in early)
    assert any("season too late" in n for n in late)
    assert any("play" in n for n in late)       # nothing left to hold for


def test_triple_captain_on_a_double_beats_the_default_prior():
    """Regression: hold EV scaled the already-doubled ep_next by the double
    multiplier again, so 'play now' was unreachable above p_double ~= 0.5 and a
    live double always read as 'hold'."""
    notes = policy.chip_advice(_boot(24), _xi(cap_ep=12.0, cap_fx=2), ["3xc"])
    assert any("Triple Captain" in n and "NOW" in n for n in notes)
    # 12.0 banked now vs 1.7 x 6.0 x 0.8 = 8.2 from a future double
    assert any("~8.2" in n for n in notes)


def test_single_gameweek_still_holds_the_triple_captain():
    notes = policy.chip_advice(_boot(10), _xi(cap_ep=6.0, cap_fx=1), ["3xc"])
    assert any("hold" in n for n in notes)
    assert not any("NOW" in n for n in notes)


def test_wildcard_advice_survives_the_default_chip_set():
    """Regression: the structural-chip loop broke after the first match, so the
    default chip set (which contains freehit) dropped all wildcard guidance."""
    notes = policy.chip_advice(_boot(10), _xi(6.0, 1),
                               ["wildcard1", "wildcard2", "bboost", "3xc", "freehit"])
    assert any("Wildcard 1" in n for n in notes)
    assert any("Free Hit" in n for n in notes)
    assert any("Triple Captain" in n for n in notes)
    assert any("Bench Boost" in n for n in notes)


def test_missing_fixture_count_does_not_claim_a_double():
    xi = _xi(6.0, 1)
    xi["captain"] = pd.Series({"web_name": "C", "ep_next": 6.0, "id": 1})  # no n_fx
    notes = policy.chip_advice(_boot(24), xi, ["3xc"])
    assert not any("on a double" in n for n in notes)


# ------------------------------------------------------------ signal roles
def _load(tmp_path, monkeypatch, body: str):
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text(body)
    return memoryio.load_signals(now=datetime(2026, 8, 20, tzinfo=timezone.utc))


def test_role_vocabulary_expands_to_minutes_bounds(tmp_path, monkeypatch):
    frame, notes = _load(tmp_path, monkeypatch, """
date: 2026-08-19
evidence:
  tier: 1
  url: https://arsenal.com/news/presser
  publisher: Arsenal.com
  published_at: 2026-08-19T09:00:00Z
adjustments:
  - player_id: 7
    role: expected_starter
  - player_id: 8
    role: not_in_predicted_xi
""")
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.85)
    assert frame.loc[8, "xmins_max"] == pytest.approx(0.35)
    assert notes[0]["applied"]


def test_unknown_role_is_rejected_loudly(tmp_path, monkeypatch):
    frame, notes = _load(tmp_path, monkeypatch, """
date: 2026-08-19
adjustments:
  - player_id: 7
    role: probably_fine
""")
    assert frame.empty
    assert any("unknown role" in p for n in notes for p in n["problems"])


def test_explicit_bound_overrides_the_role_default(tmp_path, monkeypatch):
    frame, _ = _load(tmp_path, monkeypatch, """
date: 2026-08-19
evidence:
  tier: 1
  url: https://arsenal.com/news/presser
  publisher: Arsenal.com
  published_at: 2026-08-19T09:00:00Z
adjustments:
  - player_id: 7
    role: expected_starter
    xmins_min: 0.95
""")
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.95)


def test_role_contradicting_an_explicit_cap_rejects_the_file(tmp_path, monkeypatch):
    """Regression: role expansion happened after validation, so a role floor
    above an explicit cap was accepted and both bounds were silently dropped
    instead of the file being rejected as AGENT.md promises."""
    frame, notes = _load(tmp_path, monkeypatch, """
date: 2026-08-19
adjustments:
  - player_id: 7
    role: expected_starter
    xmins_max: 0.45
    ep_per_gw: 0.5
""")
    assert frame.empty                       # nothing applied, not even the nudge
    assert not notes[0]["applied"]
    problem = " ".join(notes[0]["problems"])
    assert "contradictory bounds" in problem
    assert "expected_starter" in problem     # names the role that caused it


# ---------------------------------------------------------------- scenarios
def test_scenarios_load_and_validate(tmp_path, monkeypatch):
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text("""
date: 2026-08-19
evidence:
  tier: 2
  url: https://athletic.com/fixtures
  publisher: Athletic
  published_at: 2026-08-19T09:00:00Z
scenarios:
  - gw: 29
    kind: double
    prob: 0.7
  - gw: 99
    kind: double
    prob: 0.7
  - gw: 30
    kind: eclipse
    prob: 0.5
""")
    scen = memoryio.load_scenarios(now=datetime(2026, 8, 20, tzinfo=timezone.utc))
    assert len(scen) == 1 and scen[0]["gw"] == 29


def test_unevidenced_scenarios_cannot_steer_chip_ev(tmp_path, monkeypatch):
    """A scenario probability can FIRE a chip, so a tier-3 / no-evidence file
    supplying one would bypass every tier rule. It must load nothing."""
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text(
        "date: 2026-08-19\nscenarios:\n  - gw: 29\n    kind: double\n    prob: 0.9\n")
    assert memoryio.load_scenarios(
        now=datetime(2026, 8, 20, tzinfo=timezone.utc)) == []
