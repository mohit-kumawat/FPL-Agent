"""The evidence layer: tiers, the action gate, approvals, and shadow externals.

These pin the plan's target behaviour: a recommendation may be acted on only
when backed by official FPL data plus validated, attributable, current
evidence — and when evidence is missing or conflicting, the pipeline abstains
and names the exact research that would unblock it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from fpl_agent import (action_gate, approvals, config, digest, evidence,
                       external, memoryio, policy)

NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path / "signals")
    memoryio.SIGNALS_DIR.mkdir(parents=True)
    for name in ("CALIBRATION_FILE", "SIGNAL_LOG_FILE", "SIGNAL_SCORES_FILE",
                 "DECISIONS_FILE", "DECISION_SCORES_FILE", "RUNS_FILE"):
        monkeypatch.setattr(memoryio, name, tmp_path / name.lower())
    monkeypatch.setattr(approvals, "APPROVALS_FILE", tmp_path / "approvals.jsonl")
    monkeypatch.setattr(external, "INBOX_DIR", tmp_path / "ext" / "inbox")
    monkeypatch.setattr(external, "SNAP_DIR", tmp_path / "ext" / "snapshots")
    monkeypatch.setattr(external, "SHADOW_FILE", tmp_path / "shadow.jsonl")
    monkeypatch.setattr(digest, "CONTEXT_FILE", tmp_path / "current-context.md")
    monkeypatch.setattr(digest, "DIGEST_FILE", tmp_path / "digest.json")
    return tmp_path


def _signal(name: str, body: str) -> None:
    (memoryio.SIGNALS_DIR / name).write_text(body)


def _load():
    return memoryio.load_signals(now=NOW)


EV1 = ("evidence:\n  tier: 1\n  url: https://club.com/news/1\n"
       "  publisher: club\n  published_at: 2026-08-19T09:00:00Z\n")


# ------------------------------------------------------------ evidence tiers
def test_no_evidence_block_means_watch_only():
    """Free-text `source:` without checkable metadata cannot move anything."""
    _signal("s.yaml", "date: 2026-08-19\nsource: press conference\n"
            "adjustments:\n  - player_id: 7\n    xmins_min: 0.9\n")
    frame, notes = _load()
    assert frame.empty
    assert any("watch" in p for n in notes for p in n.get("problems", []))


def test_tier_claimed_without_url_or_timestamp_is_downgraded():
    _signal("s.yaml", "date: 2026-08-19\nevidence:\n  tier: 1\n"
            "adjustments:\n  - player_id: 7\n    xmins_min: 0.9\n")
    frame, notes = _load()
    assert frame.empty
    assert any("downgraded" in p for n in notes
               for p in n.get("evidence", {}).get("problems", []))


def test_tier_zero_cannot_be_claimed_by_a_file():
    ev = evidence.parse({"evidence": {"tier": 0, "url": "https://x.com",
                                      "published_at": "2026-08-19T09:00:00Z"}})
    assert ev["tier"] == 3
    assert any("tier 0" in p for p in ev["problems"])


def test_tier1_may_establish_hard_availability_facts():
    _signal("s.yaml", f"date: 2026-08-19\n{EV1}"
            "adjustments:\n  - player_id: 7\n    role: expected_starter\n")
    frame, _ = _load()
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.85)


def test_single_tier2_source_cannot_force_a_starter_floor():
    """A lone projected lineup is a forecast, not a fact."""
    _signal("s.yaml", "date: 2026-08-19\n"
            "evidence:\n  tier: 2\n  url: https://ffscout.com/xi\n"
            "  publisher: FFS\n  published_at: 2026-08-19T09:00:00Z\n"
            "adjustments:\n  - player_id: 7\n    role: expected_starter\n")
    frame, notes = _load()
    assert frame.empty or pd.isna(frame.loc[7].get("xmins_min"))
    assert any("second independent tier-2" in p
               for n in notes for p in n.get("problems", []))


def test_two_independent_tier2_domains_corroborate_a_hard_claim():
    for name, dom in (("a.yaml", "ffscout.com"), ("b.yaml", "athletic.com")):
        _signal(name, "date: 2026-08-19\n"
                f"evidence:\n  tier: 2\n  url: https://{dom}/xi\n"
                f"  publisher: {dom}\n  published_at: 2026-08-19T09:00:00Z\n"
                "adjustments:\n  - player_id: 7\n    role: expected_starter\n")
    frame, _ = _load()
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.85)


def test_same_domain_twice_is_not_corroboration():
    for name in ("a.yaml", "b.yaml"):
        _signal(name, "date: 2026-08-19\n"
                "evidence:\n  tier: 2\n  url: https://ffscout.com/xi\n"
                "  publisher: FFS\n  published_at: 2026-08-19T09:00:00Z\n"
                "adjustments:\n  - player_id: 7\n    role: expected_starter\n")
    frame, _ = _load()
    assert frame.empty or pd.isna(frame.loc[7].get("xmins_min"))


def test_soft_forecast_roles_need_only_a_single_tier2_source():
    _signal("s.yaml", "date: 2026-08-19\n"
            "evidence:\n  tier: 2\n  url: https://athletic.com/rotation\n"
            "  publisher: Athletic\n  published_at: 2026-08-19T09:00:00Z\n"
            "adjustments:\n  - player_id: 7\n    role: rotation_risk\n")
    frame, _ = _load()
    assert frame.loc[7, "xmins_max"] == pytest.approx(0.70)


def test_evidence_lifetime_caps_the_signal_expiry():
    """A tier-2 claim published 10 days ago is stale even if ttl_days says 30."""
    _signal("s.yaml", "date: 2026-08-10\nttl_days: 30\n"
            "evidence:\n  tier: 2\n  url: https://athletic.com/x\n"
            "  publisher: A\n  published_at: 2026-08-10T09:00:00Z\n"
            "adjustments:\n  - player_id: 7\n    role: rotation_risk\n")
    frame, notes = _load()
    assert frame.empty
    assert any("expired" in p for n in notes for p in n.get("problems", []))


def test_cross_file_conflict_resolves_to_the_higher_tier():
    _signal("club.yaml", f"date: 2026-08-19\n{EV1}"
            "adjustments:\n  - player_id: 7\n    xmins_min: 0.9\n")
    _signal("outlet.yaml", "date: 2026-08-19\n"
            "evidence:\n  tier: 2\n  url: https://athletic.com/x\n"
            "  publisher: A\n  published_at: 2026-08-19T09:00:00Z\n"
            "adjustments:\n  - player_id: 7\n    xmins_max: 0.4\n")
    frame, notes = _load()
    assert frame.loc[7, "xmins_min"] == pytest.approx(0.9)   # tier 1 floor kept
    assert pd.isna(frame.loc[7, "xmins_max"])                # tier 2 cap dropped
    assert any("higher evidence tier" in p
               for n in notes for p in n.get("problems", []))


def test_equal_tier_conflict_drops_both_bounds():
    for name, adj in (("a.yaml", "xmins_min: 0.9"), ("b.yaml", "xmins_max: 0.4")):
        _signal(name, "date: 2026-08-19\n"
                f"evidence:\n  tier: 2\n  url: https://{name[0]}site.com/x\n"
                f"  publisher: {name}\n  published_at: 2026-08-19T09:00:00Z\n"
                f"adjustments:\n  - player_id: 7\n    {adj}\n")
    frame, notes = _load()
    assert pd.isna(frame.loc[7, "xmins_min"]) and pd.isna(frame.loc[7, "xmins_max"])
    assert any("both minutes bounds dropped" in p
               for n in notes for p in n.get("problems", []))


# ------------------------------------------------------------- the action gate
def _players(rows: list[dict]) -> pd.DataFrame:
    base = {"status": "a", "play_chance": 1.0, "web_name": "?", "ep_next": 5.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _transfer_ctx(in_ids: list[int], players: pd.DataFrame) -> dict:
    plan = {"n_transfers": len(in_ids), "net_gain_vs_hold": 7.0, "hit_cost": 0,
            "in": players[players["id"].isin(in_ids)],
            "out": players.iloc[0:0]}
    return {"recommendation": {"transfers": {"action": "1_transfer", "plan": plan}},
            "signal_notes": []}


def test_transfer_with_clean_incoming_qualifies():
    players = _players([{"id": 1, "web_name": "Fit"}])
    gate = action_gate.evaluate(_transfer_ctx([1], players), players,
                                {"players": []}, None)
    assert gate["status"] == "qualified"
    assert gate["failed_requirements"] == []


def test_transfer_blocked_when_incoming_availability_is_unverified():
    players = _players([{"id": 1, "web_name": "Doubt", "status": "d",
                         "play_chance": 0.5}])
    gate = action_gate.evaluate(_transfer_ctx([1], players), players,
                                {"players": []}, None)
    assert gate["status"] == "blocked"
    assert any("Doubt" in f for f in gate["failed_requirements"])
    assert gate["next_research"], "a block must name the unblocking research"
    assert "BLOCKED — NO ACTIONABLE TRANSFER" in action_gate.headline(gate)


def test_an_applied_signal_floor_is_availability_evidence():
    players = _players([{"id": 1, "web_name": "Doubt", "status": "d",
                         "play_chance": 0.5}])
    signals = pd.DataFrame({"xmins_min": [0.85], "xmins_max": [None]}, index=[1])
    gate = action_gate.evaluate(_transfer_ctx([1], players), players,
                                {"players": []}, signals)
    assert gate["status"] == "qualified"


def test_the_deadlock_resolves_to_owner_choice():
    """Transfer lacks evidence AND holding carries an unresolved owned risk:
    the gate must hand over both sides, not fall silent."""
    players = _players([
        {"id": 1, "web_name": "DoubtIn", "status": "d", "play_chance": 0.5},
        {"id": 2, "web_name": "InjuredOwned", "status": "i", "play_chance": 0.1},
    ])
    gate = action_gate.evaluate(_transfer_ctx([1], players), players,
                                {"players": [{"id": 2}]}, None)
    assert gate["status"] == "owner_choice"
    assert any("DoubtIn" in f for f in gate["failed_requirements"])
    assert any("holding is not risk-free" in f for f in gate["failed_requirements"])


def test_hold_with_unresolved_owned_risk_is_owner_choice_not_silence():
    players = _players([{"id": 2, "web_name": "InjuredOwned", "status": "i",
                         "play_chance": 0.1}])
    ctx = {"recommendation": {"transfers": {"action": "hold", "plan": None}},
           "signal_notes": []}
    gate = action_gate.evaluate(ctx, players, {"players": [{"id": 2}]}, None)
    assert gate["status"] == "owner_choice"
    assert gate["next_research"]


def test_clean_hold_qualifies_without_research():
    players = _players([{"id": 2, "web_name": "Fit"}])
    ctx = {"recommendation": {"transfers": {"action": "hold", "plan": None}},
           "signal_notes": []}
    gate = action_gate.evaluate(ctx, players, {"players": [{"id": 2}]}, None)
    assert gate["status"] == "qualified"


def test_captain_band_without_evidence_is_owner_choice():
    """The pick is fit, but a candidate within 1.0 EP is flagged — the call is
    the owner's until the flag is researched."""
    players = _players([
        {"id": 1, "web_name": "Pick", "ep_next": 8.0},
        {"id": 2, "web_name": "CloseDoubt", "ep_next": 7.5, "status": "d",
         "play_chance": 0.5},
    ])
    xi = {"xi": players, "captain": players.iloc[0]}
    ctx = {"recommendation": {"transfers": {"action": "hold", "plan": None},
                              "xi": xi}, "signal_notes": []}
    gate = action_gate.evaluate(ctx, players, {"players": []}, None)
    assert gate["captain"]["status"] == "owner_choice"
    assert gate["status"] == "owner_choice"      # overall inherits the doubt


def test_unresolved_signal_conflict_blocks_the_touched_transfer():
    players = _players([{"id": 1, "web_name": "Target"}])
    ctx = _transfer_ctx([1], players)
    ctx["signal_notes"] = [{"file": "(merged)", "applied": False,
                            "conflict_player": 1,
                            "problems": ["player 1: floor above a cap"]}]
    gate = action_gate.evaluate(ctx, players, {"players": []}, None,
                                ctx["signal_notes"])
    assert gate["status"] == "blocked"
    assert any("conflict" in f for f in gate["failed_requirements"])


# --------------------------------------------------------------- chip evidence
def _xi(cap_ep: float, cap_fx: int, bench_ep: float = 6.0) -> dict:
    return {"captain": pd.Series({"web_name": "P", "ep_next": cap_ep,
                                  "next_n_fixtures": cap_fx, "id": 1}),
            "xi": pd.DataFrame([{"id": 1, "web_name": "P", "ep_next": cap_ep}]),
            "vice": pd.Series({"web_name": "V", "id": 2}),
            "bench_order": pd.DataFrame([{"id": 3, "ep_next": bench_ep}])}


def _boot_gw(gw: int) -> dict:
    return {"events": [{"id": gw, "is_next": True, "is_current": False,
                        "finished": False}]}


def test_default_prior_alone_never_fires_a_chip():
    """Clears the bar at the assumed p=0.8 but not against a certain double:
    that is exactly the scenario-dependent case the owner must decide."""
    # now=10.0 on a double (per-fx 5.0): hold(0.8)=6.8 -> clears (bar 7.82);
    # hold(1.0)=8.5 -> bar 9.78 ... 10 >= 9.78 clears. Use now=9.0:
    # hold(0.8)=6.12 bar 7.04 -> clears; hold(1.0)=7.65 bar 8.8 -> 9.0 clears too.
    # Single fixture makes it easy: now=8.0, hold(0.8)=8*1.7*0.8=10.9 -> holds.
    # Need now between 1.15*hold(.8) and 1.15*hold(1.0): fx=2, per-fx=now/2,
    # hold(.8)=.68*now -> bar .78*now (always clears); hold(1)=.85*now ->
    # bar .978*now (always clears). 3xc on doubles always dominates; use bboost:
    # bench now=10, hold(.8)=1.8*.8*10=14.4 holds. bench per-fx n=1: hold scales
    # with same sum — construct with mixed: bench ep 10 single fixtures:
    # hold(0.8) = 10*1.8*0.8 = 14.4 -> hold. Not reachable either. The reachable
    # window needs a double: bench ep 10 across doubles (per-fx 5):
    # hold(.8)=7.2 bar 8.28 -> 10 clears; hold(1)=9 bar 10.35 -> 10 does NOT.
    xi = _xi(0.0, 1)
    xi["bench_order"] = pd.DataFrame(
        [{"id": 3, "ep_next": 10.0, "next_n_fixtures": 2}])
    notes = policy.chip_advice(_boot_gw(10), xi, ["bboost1"])
    assert any("OWNER CHOICE" in n and "Bench Boost" in n for n in notes)
    assert not any("NOW" in n for n in notes)


def test_researched_scenario_probability_still_fires_a_chip():
    xi = _xi(0.0, 1)
    xi["bench_order"] = pd.DataFrame(
        [{"id": 3, "ep_next": 10.0, "next_n_fixtures": 2}])
    scen = [{"gw": 29, "kind": "double", "prob": 0.8}]
    notes = policy.chip_advice(_boot_gw(10), xi, ["bboost1"], scenarios=scen)
    assert any("Bench Boost" in n and "NOW" in n for n in notes)


def test_dominant_chip_play_survives_the_evidence_rule():
    """A play that wins even against a CERTAIN future double needs no research —
    no scenario answer could change it."""
    notes = policy.chip_advice(_boot_gw(24), _xi(cap_ep=12.0, cap_fx=2), ["3xc"])
    assert any("Triple Captain" in n and "NOW" in n for n in notes)


def test_gate_reads_chip_owner_choice_out_of_the_notes():
    ctx = {"recommendation": {"transfers": {"action": "hold", "plan": None},
                              "chips": ["[MODEL] Bench Boost 1: ... OWNER CHOICE "
                                        "pending scenario research ..."]},
           "signal_notes": []}
    players = _players([])
    gate = action_gate.evaluate(ctx, players, {"players": []}, None)
    assert gate["chips"]["status"] == "owner_choice"
    assert any("scenario" in r for r in gate["next_research"])


# ------------------------------------------------------------------ approvals
def _decision(action="1_transfer", captain="Salah") -> dict:
    return {"date": "2026-08-20", "gw": 2, "action": action,
            "in": ["Gyokeres"], "out": ["Watkins"], "captain": captain}


def test_identical_proposals_collapse_into_one_awaiting_decision():
    for _ in range(5):
        approvals.record_proposal(_decision())
    events = [json.loads(l) for l in
              approvals.APPROVALS_FILE.read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "proposed"]) == 1
    assert approvals.state()["awaiting_owner"]


def test_owner_decision_moves_the_lifecycle():
    approvals.record_proposal(_decision())
    approvals.record_owner_decision("approved", "go for it")
    assert approvals.state()["state"] == "approved"


def test_rejected_proposal_stays_visible_until_superseded():
    approvals.record_proposal(_decision())
    approvals.record_owner_decision("rejected", "too risky")
    assert approvals.state()["state"] == "rejected"
    approvals.record_proposal(_decision(captain="Haaland"))   # new proposal
    st = approvals.state()
    assert st["state"] == "proposed" and st["awaiting_owner"]


def test_approval_is_not_execution_until_official_picks_reconcile(monkeypatch):
    approvals.record_proposal(_decision())
    approvals.record_owner_decision("approved")
    squad = {"entry_id": 42,
             "players": [{"id": 10, "name": "Gyokeres"}]}
    boot = {"events": [{"id": 2, "is_current": True}],
            "elements": [{"id": 10, "web_name": "Gyokeres"}]}

    # official picks do NOT contain the incoming player yet
    monkeypatch.setattr(approvals.data, "fetch_picks",
                        lambda e, g: {"picks": [{"element": 99}]})
    res = approvals.try_reconcile(squad, boot)
    assert res and not res["ok"]
    assert approvals.state()["state"] == "approved", "still not executed"

    # now they do
    monkeypatch.setattr(approvals.data, "fetch_picks",
                        lambda e, g: {"picks": [{"element": 10}]})
    res = approvals.try_reconcile(squad, boot)
    assert res and res["ok"]
    assert approvals.state()["state"] == "reconciled"


def test_reconcile_survives_network_failure():
    approvals.record_proposal(_decision())
    approvals.record_owner_decision("approved")
    # no entry_id -> no fetch attempted, no crash, state unchanged
    assert approvals.try_reconcile({"players": []}, {"events": []}) is None
    assert approvals.state()["state"] == "approved"


def test_approvals_module_never_touches_squad_yaml(tmp_path, monkeypatch):
    squad_file = tmp_path / "squad.yaml"
    squad_file.write_text("players: []\n")
    before = squad_file.read_text()
    monkeypatch.setattr(config, "SQUAD_FILE", squad_file)
    approvals.record_proposal(_decision())
    approvals.record_owner_decision("approved")
    assert squad_file.read_text() == before


# ------------------------------------------------------------ shadow externals
PAYLOAD = {"provider": "oddsfeed", "version": "1", "gw": 5,
           "probabilities": [{"team_h": "ARS", "team_a": "CHE",
                              "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15,
                              "p_cs_home": 0.6}]}
BOOT = {"teams": [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "CHE"}],
        "events": [{"id": 5, "finished": True}]}
FIXTURES = [{"event": 5, "finished": True, "team_h": 1, "team_a": 2,
             "team_h_score": 2, "team_a_score": 0}]


def _drop_payload(payload: dict, name: str = "odds.json") -> None:
    external.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    (external.INBOX_DIR / name).write_text(json.dumps(payload))


def test_shadow_ingest_snapshots_and_scores_against_results():
    _drop_payload(PAYLOAD)
    out = external.run_shadow(BOOT, FIXTURES)
    assert out["ingested"][0]["ingested"]
    assert not list(external.INBOX_DIR.glob("*.json")), "inbox drained"
    rec = out["scored"][0]
    # home win with p=0.6: brier = (0.4^2 + 0.25^2 + 0.15^2)
    assert rec["brier"] == pytest.approx(0.245, abs=1e-3)
    assert rec["cs_accuracy"] == 1.0
    assert external.shadow_summary()["oddsfeed"]["gws_scored"] == 1
    assert not external.shadow_summary()["oddsfeed"]["promotable"]


def test_malformed_payload_fails_closed_and_stays_visible():
    _drop_payload({"provider": "bad", "gw": 5, "probabilities": [
        {"team_h": "ARS", "team_a": "CHE", "p_home": 0.9, "p_draw": 0.9,
         "p_away": 0.9}]})
    out = external.run_shadow(BOOT, FIXTURES)
    assert out["ingested"][0]["problems"]
    assert list(external.INBOX_DIR.glob("*.json")), "left in inbox for a human"
    assert out["scored"] == []


def test_shadow_never_scores_the_same_provider_gw_twice():
    _drop_payload(PAYLOAD)
    assert len(external.run_shadow(BOOT, FIXTURES)["scored"]) == 1
    _drop_payload(PAYLOAD, "again.json")
    assert external.run_shadow(BOOT, FIXTURES)["scored"] == []


def test_shadow_data_is_invisible_to_expected_points():
    """The isolation guarantee, structurally: nothing in the model layer
    imports the external adapter."""
    import fpl_agent.models as m
    import fpl_agent.optimizer as o
    src = open(m.__file__).read() + open(o.__file__).read()
    assert "external" not in src


# ------------------------------------------------------- context integration
def test_digest_carries_gate_approvals_and_stays_bounded():
    approvals.record_proposal(_decision())
    for gw in range(1, 39):
        memoryio.log_calibration({"gw": gw, "mae_all": 2.0, "mae_top50": 3.0, "n": 500})
        memoryio.log_decision({"date": f"w{gw}", "action": "hold",
                               "captain": "X", "gate": "qualified", "gw": gw})
    gate = {"status": "blocked", "action_type": "transfer",
            "failed_requirements": ["incoming availability unverified: Doubt"],
            "next_research": ["resolve: Doubt flagged — find a tier-1 source"],
            "captain": {"status": "qualified", "unevidenced": []},
            "chips": {"status": "qualified"}}
    boot = {"events": [{"id": 10, "is_next": True,
                        "deadline_time": "2026-10-01T11:00:00Z"}], "chips": []}
    squad = {"players": [{"id": i} for i in range(15)], "bank": 1.0,
             "free_transfers": 2, "chips_available": []}
    d = digest.build(boot, [], squad, {}, gate=gate)
    text = digest.render(d)
    assert "Latest gate" in text and "BLOCKED" in text
    assert "Needs you" in text and "fpl approve" in text
    assert "Research queue" in text
    assert "Gate record" in text
    assert len(text) < 4000, f"context grew to {len(text)} bytes"


def test_captain_regret_is_scored_from_the_decision_time_xi():
    memoryio.log_decision({"date": "d", "gw": 5, "action": "hold",
                           "captain": "Pick", "xi_names": ["Pick", "Better"]})
    panel = pd.DataFrame([{"element": 1, "total_points": 2, "minutes": 90},
                          {"element": 2, "total_points": 12, "minutes": 90}])
    rec = memoryio.score_captaincy(5, panel, {1: "Pick", 2: "Better"})
    assert rec["regret"] == pytest.approx(10.0)
    assert memoryio.score_captaincy(5, panel, {1: "Pick", 2: "Better"}) is None
    dq = memoryio.decision_quality()
    assert dq["captain_regret_mean"] == pytest.approx(10.0)
