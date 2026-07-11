"""Strategy ("base_eq") equity-curve persistence.

Mirrors the equity curve the RsDynamicAse.pine indicator plots: the *simulated*
strategy equity compounding the selected assets' daily returns, coloured by the
held asset per bar.  This is the theoretical strategy performance, NOT the real
account value (that lives in ``equity.py``).

Design — accurate but cheap:
  The curve is fully deterministic — it is recomputed from immutable daily closes
  plus the same signals used live (signals.engine → backtest.run_equity_simulation)
  on every signal cycle.  So we do NOT store the curve point-by-point.  The ONLY
  durable state is a single ``anchor`` date — the day the feature was first enabled
  ("today") — kept in state/strategy_equity_anchor.json on the fly volume.  Every
  recompute re-derives the identical curve from the anchor forward and normalises
  it to 1.0 at the anchor, so the result survives restarts/redeploys unchanged.

  A cached payload (state/strategy_equity.json) is also written so the dashboard
  has data instantly after a restart and even if a later compute cycle fails — but
  it is a cache, never the source of truth.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ANCHOR_FILE = Path("state/strategy_equity_anchor.json")
_CURVE_FILE = Path("state/strategy_equity.json")


def _get_or_set_anchor(latest_date: str) -> str:
    """Return the persisted anchor date, seeding it to *latest_date* on first use.

    The anchor is the curve's "start from today" point. It is written once and
    never changed thereafter, so the curve always begins on the same day.
    """
    if _ANCHOR_FILE.exists():
        try:
            anchor = json.loads(_ANCHOR_FILE.read_text(encoding="utf-8")).get("anchor")
            if anchor:
                return str(anchor)
        except Exception as exc:
            logger.warning("Could not read strategy-equity anchor (%s) — reseeding.", exc)
    try:
        _ANCHOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ANCHOR_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"anchor": latest_date}), encoding="utf-8")
        os.replace(tmp, _ANCHOR_FILE)
        logger.info("Strategy-equity anchor seeded at %s.", latest_date)
    except Exception as exc:
        logger.error("Failed to persist strategy-equity anchor: %s", exc)
    return latest_date


def build_payload(curve: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Slice the full simulated curve from the anchor and normalise to 1.0 there.

    *curve* is the full daily series from the engine: ``[{date, equity, asset}]``.
    Returns the dashboard payload ``{anchor, current, max_dd, points:[{ts, equity,
    asset}]}`` or None when there's nothing to show.
    """
    if not curve:
        return None
    anchor = _get_or_set_anchor(curve[-1]["date"])

    pts = [p for p in curve if p.get("date", "") >= anchor]
    if not pts:  # anchor newer than every bar (shouldn't happen) — show last point
        pts = [curve[-1]]

    base = pts[0].get("equity") or 1.0
    if base <= 0:
        base = 1.0

    norm: List[Dict[str, Any]] = []
    peak = 0.0
    max_dd = 0.0
    for p in pts:
        v = (p.get("equity") or 0.0) / base
        norm.append({"ts": p["date"], "equity": round(v, 6), "asset": p.get("asset")})
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    return {
        "anchor": anchor,
        "current": norm[-1]["equity"],
        "max_dd": round(max_dd, 4),
        "points": norm,
    }


def write_curve(curve: List[Dict[str, Any]]) -> None:
    """Build the anchored payload from *curve* and cache it atomically.

    Never raises — a cache-write failure must not abort a trade cycle.
    """
    try:
        payload = build_payload(curve)
        if payload is None:
            return
        _CURVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CURVE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, _CURVE_FILE)
        logger.info(
            "Strategy equity cached → %s (%d points since %s, current %.4fx).",
            _CURVE_FILE, len(payload["points"]), payload["anchor"], payload["current"],
        )
    except Exception as exc:
        logger.error("Failed to cache strategy equity: %s", exc)


def read_payload() -> Optional[Dict[str, Any]]:
    """Return the most recently cached strategy-equity payload, or None."""
    if not _CURVE_FILE.exists():
        return None
    try:
        return json.loads(_CURVE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read cached strategy equity: %s", exc)
        return None
