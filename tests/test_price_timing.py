"""Price-change predictor notes: timing advice only, never selection."""
from __future__ import annotations

import pandas as pd

from fpl_agent import policy


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "web_name": "Riser", "price": 7.5, "price_change_percent": 95.0},
        {"id": 2, "web_name": "Momentum", "price": 6.0, "price_change_percent": 70.0},
        {"id": 3, "web_name": "Faller", "price": 8.0, "price_change_percent": -92.0},
        {"id": 4, "web_name": "Quiet", "price": 5.0, "price_change_percent": 0.0},
    ])


def test_imminent_rise_on_a_target_says_act_tonight():
    notes = policy.price_timing_notes(_players(), in_ids=[1], out_ids=[], owned_ids=[])
    assert len(notes) == 1 and "RISE" in notes[0] and "before 00:00 UK" in notes[0]


def test_momentum_without_imminence_is_calm():
    notes = policy.price_timing_notes(_players(), in_ids=[2], out_ids=[], owned_ids=[])
    assert len(notes) == 1 and "no urgency" in notes[0]


def test_falling_target_says_wait_for_the_discount():
    notes = policy.price_timing_notes(_players(), in_ids=[3], out_ids=[], owned_ids=[])
    assert len(notes) == 1 and "FALL" in notes[0] and "cheaper" in notes[0]


def test_falling_outgoing_says_sell_before_the_boundary():
    notes = policy.price_timing_notes(_players(), in_ids=[], out_ids=[3], owned_ids=[3])
    assert len(notes) == 1 and "protects the selling price" in notes[0]


def test_owned_faller_not_being_sold_is_flagged_without_forcing_action():
    notes = policy.price_timing_notes(_players(), in_ids=[], out_ids=[], owned_ids=[3, 4])
    assert len(notes) == 1 and "value at risk" in notes[0] and "no action forced" in notes[0]


def test_quiet_market_produces_no_notes():
    notes = policy.price_timing_notes(_players(), in_ids=[4], out_ids=[4], owned_ids=[4])
    assert notes == []


def test_missing_column_is_silent():
    notes = policy.price_timing_notes(_players().drop(columns=["price_change_percent"]),
                                      in_ids=[1], out_ids=[], owned_ids=[])
    assert notes == []
