"""Evidence tiers for research signals: who said it, and what that permits.

The blind-label eval priced the agent's *opinions* at negative value; the fix
was restricting signals to minutes FACTS. This module closes the remaining gap:
a "fact" was whatever the YAML said it was. Now every signal file names its
evidence, and what a file may DO follows from the tier of that evidence:

  tier 1   club / manager / league official        may establish facts
  tier 2   named journalist / established outlet   forecasts; hard facts only
                                                   when two independent tier-2
                                                   domains agree
  tier 3   aggregator / anonymous / no metadata    watch item only — never moves
                                                   minutes or EP

(Tier 0 is the official FPL API itself. A YAML file cannot claim tier 0 — the
pipeline already reads the API directly, so a hand-written copy of it would only
add a place for the copy to be wrong.)

A file with no `evidence` block, or with a tier it cannot substantiate (no URL,
no timestamp), is treated as tier 3. It is still parsed and still reported —
silently dropping research is how contradictions hide — it just cannot steer
a recommendation.

Hard roles (`ruled_out`, `expected_starter`) are availability claims that gate
actions downstream, so they get the strictest rule: tier 1, or two tier-2
sources from different domains making the same claim about the same player.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

TIER_LABELS = {
    1: "official (club/manager/league)",
    2: "named journalist / established outlet",
    3: "aggregator / unverified",
}
# Roles whose bounds assert availability facts, not rotation forecasts.
HARD_ROLES = ("ruled_out", "expected_starter")
# A source is only current for so long: a presser quote ages differently from
# an aggregator guess. The signal's own expiry is capped to published_at + ttl.
TIER_MAX_TTL_DAYS = {1: 14, 2: 7, 3: 3}
DEFAULT_TIER = 3


def domain(url: str | None) -> str | None:
    """Registrable host of a URL, lowercased, `www.` stripped. None if absent."""
    if not url:
        return None
    try:
        host = (urlparse(str(url)).netloc or "").lower()
    except ValueError:
        return None
    return host.removeprefix("www.") or None


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def parse(doc: dict) -> dict:
    """Normalise a signal file's `evidence` block.

    Returns {tier, url, domain, publisher, published_at, problems}. A tier the
    metadata cannot substantiate is DOWNGRADED to 3 rather than rejected: the
    file stays visible in reports, it just loses the right to move anything.
    """
    ev = doc.get("evidence") or {}
    problems: list[str] = []
    if not ev:
        return {"tier": DEFAULT_TIER, "url": None, "domain": None,
                "publisher": None, "published_at": None,
                "problems": ["no evidence block — treated as tier 3 (watch only)"]}

    try:
        tier = int(ev.get("tier", DEFAULT_TIER))
    except (TypeError, ValueError):
        tier = DEFAULT_TIER
        problems.append(f"evidence.tier={ev.get('tier')!r} is not an integer — tier 3 assumed")
    if tier == 0:
        tier = DEFAULT_TIER
        problems.append("tier 0 is the FPL API itself — a YAML file cannot claim it; tier 3 assumed")
    elif tier not in TIER_LABELS:
        problems.append(f"unknown tier {tier} — tier 3 assumed")
        tier = DEFAULT_TIER

    url = ev.get("url")
    dom = domain(url)
    published_at = _parse_ts(ev.get("published_at"))

    if tier <= 2:
        missing = [k for k, v in (("url", dom), ("published_at", published_at)) if not v]
        if missing:
            problems.append(f"tier {tier} claimed without {'/'.join(missing)} — "
                            "downgraded to tier 3 (a source that cannot be checked "
                            "cannot establish anything)")
            tier = DEFAULT_TIER

    return {"tier": tier, "url": url, "domain": dom,
            "publisher": ev.get("publisher"), "published_at": published_at,
            "problems": problems}


def effective_expiry(ev: dict, file_expiry: datetime | None) -> datetime | None:
    """The signal's expiry, capped to the source's permitted lifetime."""
    pub = ev.get("published_at")
    if pub is None:
        return file_expiry
    cap = pub + timedelta(days=TIER_MAX_TTL_DAYS[ev["tier"]])
    return min(file_expiry, cap) if file_expiry else cap


def may_apply_bounds(ev: dict, role: str | None, corroborated: bool) -> tuple[bool, str | None]:
    """May this file's minutes bounds for this adjustment steer the model?

    Returns (allowed, reason-if-not). `corroborated` means a second tier<=2 file
    from a DIFFERENT domain makes the same hard claim about the same player.
    """
    tier = ev["tier"]
    if tier >= 3:
        return False, ("tier 3 / unverifiable — watch item only; needs a club or "
                       "named-outlet source with url + published_at")
    if role in HARD_ROLES and tier == 2 and not corroborated:
        return False, (f"'{role}' is an availability fact — needs tier 1 evidence "
                       "or a second independent tier-2 source")
    return True, None


def may_apply_ep(ev: dict) -> bool:
    """ep_per_gw nudges are forecasts: tier <= 2 only."""
    return ev["tier"] <= 2
