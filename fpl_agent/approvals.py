"""Owner approvals: the formal record of what was proposed, decided, and done.

The pipeline recommends; the OWNER acts on the FPL site. Before this module the
gap between those two was informal — a recommendation appeared in a report, and
whether it was accepted could only be inferred from a later squad.yaml edit.
This ledger makes each step an explicit event:

  proposed     the pipeline logged a recommendation (one event per distinct
               proposal, not one per run — a quiet week repeats itself)
  approved /   the owner's call, recorded via `fpl approve`. Rejected and
  rejected /   deferred proposals stay visible until superseded.
  deferred
  reconciled   the approved action is VISIBLE IN THE OFFICIAL PICKS. Approval
               is intent; only reconciliation is execution. The FPL API is the
               source of truth for what actually happened.

Rules the rest of the pipeline honours:
  * recommendations NEVER update squad.yaml — that file changes only after the
    owner acts and confirms;
  * this module never mutates squad state either: it is a ledger, not an actor.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from . import config, data

APPROVALS_FILE = config.MEMORY_DIR / "approvals.jsonl"
OWNER_DECISIONS = ("approved", "rejected", "deferred")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read() -> list[dict]:
    if not APPROVALS_FILE.exists():
        return []
    out = []
    for line in APPROVALS_FILE.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append(event: dict) -> None:
    with APPROVALS_FILE.open("a") as fh:
        fh.write(json.dumps({"ts": _now(), **event}) + "\n")


def rec_id(decision: dict) -> str:
    """Stable id of a proposal: what would change hands, not when it was said.

    Hashing the action's CONTENT (players, captain — and for an initial build,
    the whole squad) and NOT the date is what lets a quiet week of identical
    recommendations collapse into one proposal awaiting one owner decision,
    instead of a new approval request every run. The squad list must be in the
    hash: without it, two different proposed 15s with the same captain would
    collide and the second would silently never be proposed.
    """
    payload = {k: decision.get(k)
               for k in ("action", "in", "out", "squad", "captain", "gw")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def record_proposal(decision: dict) -> str:
    """Log a proposal event unless the latest proposal is already identical."""
    rid = rec_id(decision)
    last = next((e for e in reversed(_read()) if e.get("kind") == "proposed"), None)
    if last is None or last.get("rec_id") != rid:
        _append({"kind": "proposed", "rec_id": rid,
                 "gw": decision.get("gw"), "action": decision.get("action"),
                 "in": decision.get("in"), "out": decision.get("out"),
                 "squad": decision.get("squad"),
                 "captain": decision.get("captain")})
    return rid


def record_owner_decision(decision: str, note: str = "",
                          rid: str | None = None) -> dict:
    """Record the owner's call on a proposal (latest one when rid is omitted)."""
    if decision not in OWNER_DECISIONS:
        raise ValueError(f"decision must be one of {OWNER_DECISIONS}")
    target = rid
    if target is None:
        last = next((e for e in reversed(_read()) if e.get("kind") == "proposed"), None)
        if last is None:
            raise ValueError("no proposal on record to decide on")
        target = last["rec_id"]
    event = {"kind": "owner", "rec_id": target, "decision": decision, "note": note}
    _append(event)
    return event


def record_reconciliation(rid: str, ok: bool, detail: str) -> None:
    # a still-pending reconciliation is retried every daily run — dedupe the
    # unchanged outcome or the ledger fills with identical failure events
    last = next((e for e in reversed(_read())
                 if e.get("kind") == "reconciled" and e.get("rec_id") == rid), None)
    if last and last.get("ok") == ok and last.get("detail") == detail:
        return
    _append({"kind": "reconciled", "rec_id": rid, "ok": ok, "detail": detail})


def state() -> dict:
    """Lifecycle of the latest proposal: proposed → owner call → reconciled.

    Never talks to the network — pure ledger read, safe for the digest.
    """
    events = _read()
    prop = next((e for e in reversed(events) if e.get("kind") == "proposed"), None)
    if prop is None:
        return {"state": "none"}
    rid = prop["rec_id"]
    owner = next((e for e in reversed(events)
                  if e.get("kind") == "owner" and e.get("rec_id") == rid), None)
    recon = next((e for e in reversed(events)
                  if e.get("kind") == "reconciled" and e.get("rec_id") == rid
                  and e.get("ok")), None)
    st = "proposed"
    if owner:
        st = owner["decision"]
    if recon and owner and owner["decision"] == "approved":
        st = "reconciled"
    return {"state": st, "rec_id": rid, "gw": prop.get("gw"),
            "action": prop.get("action"), "captain": prop.get("captain"),
            "in": prop.get("in"), "out": prop.get("out"),
            "proposed_at": prop.get("ts"),
            "owner_note": (owner or {}).get("note"),
            "awaiting_owner": owner is None}


def try_reconcile(squad: dict, boot: dict) -> dict | None:
    """If the latest proposal is approved but not yet reconciled, check the
    official picks for it. Network is best-effort: failure returns None and
    changes nothing — reconciliation simply retries next run.

    A transfer reconciles when every incoming player appears in the official
    picks for a gameweek at or after the proposal's target. A hold/initial
    proposal reconciles when squad.yaml matches the official picks (verify.py
    already blocks on that mismatch, so official presence is the check).
    """
    st = state()
    if st.get("state") != "approved":
        return None
    entry_id = squad.get("entry_id")
    cur = next((e for e in (boot or {}).get("events", []) if e.get("is_current")), None)
    if not entry_id or not cur:
        return None
    try:
        picks = data.fetch_picks(int(entry_id), int(cur["id"]))
        official = {p["element"] for p in picks.get("picks", [])}
    except Exception:  # noqa: BLE001 — network best-effort by design
        return None
    if not official:
        return None

    in_names = st.get("in") or []
    if in_names:
        # proposals store FPL web_names, so map back through the SAME source
        # (bootstrap elements) — squad.yaml names are hand-typed and one accent
        # of drift would stall reconciliation forever. Duplicate web_names
        # (two Wards) are ambiguous and excluded rather than guessed.
        counts: dict[str, int] = {}
        for el in (boot or {}).get("elements", []):
            counts[el["web_name"]] = counts.get(el["web_name"], 0) + 1
        name_to_id = {el["web_name"]: int(el["id"])
                      for el in (boot or {}).get("elements", [])
                      if counts.get(el["web_name"]) == 1}
        ids = [name_to_id.get(n) for n in in_names]
        if any(i is None for i in ids):
            record_reconciliation(st["rec_id"], False,
                                  f"cannot map {in_names} to unambiguous element "
                                  "ids — reconcile by eye and update squad.yaml")
            return None
        ok = all(i in official for i in ids)
        detail = (f"incoming {in_names} present in official GW{cur['id']} picks"
                  if ok else f"incoming {in_names} not in official picks yet")
    else:
        local = {int(p["id"]) for p in (squad.get("players") or []) if p.get("id")}
        ok = bool(local) and local == official
        detail = ("squad.yaml matches official picks" if ok
                  else "squad.yaml does not match official picks")
    record_reconciliation(st["rec_id"], ok, detail)
    return {"ok": ok, "detail": detail, "rec_id": st["rec_id"]}
