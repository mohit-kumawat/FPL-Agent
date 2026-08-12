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
    assert any("Triple Captain NOW" in n for n in notes)


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


# ------------------------------------------------------------ signal roles
def _load(tmp_path, monkeypatch, body: str):
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text(body)
    return memoryio.load_signals(now=datetime(2026, 8, 20, tzinfo=timezone.utc))


def test_role_vocabulary_expands_to_minutes_bounds(tmp_path, monkeypatch):
    frame, notes = _load(tmp_path, monkeypatch, """
date: 2026-08-19
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
adjustments:
  - player_id: 7
    role: expected_starter
    xmins_min: 0.95
""")
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.95)


# ---------------------------------------------------------------- scenarios
def test_scenarios_load_and_validate(tmp_path, monkeypatch):
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text("""
date: 2026-08-19
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
