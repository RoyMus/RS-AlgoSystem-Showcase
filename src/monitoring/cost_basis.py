"""Per-asset cost-basis ledger → true unrealized P&L for the position monitor.

The rest of the system only ever knows a position's *current* value, never what it
cost to acquire. This ledger fills that gap so the Telegram profit monitor can ask
"SOL is up +27%, trim it?" rather than just "SOL is overweight".

Model — tracked per BASE currency, portfolio-wide (all clients run the same strategy
and rebalance together, so a single portfolio-level average entry is accurate enough
and far simpler/robuster than a per-client ledger):

    { "<BASE>": {"amount": float, "quote_invested": float} }
    avg_entry = quote_invested / amount

Maintenance:
  - apply_orders()  — fold real fills into the running average (buys raise the basis,
                      sells reduce amount/invested proportionally; avg cost is unchanged
                      by a sale).
  - reconcile()     — clamp the ledger to reality after each snapshot: drop closed
                      positions, scale down when fees/dust left less than we think.
  - seed_missing()  — adopt an unledgered holding (pre-existing or externally deposited)
                      at the current price so it only alerts after genuinely growing.

Persisted atomically to state/cost_basis.json on the fly volume, so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from ..models import OrderResult

logger = logging.getLogger(__name__)

_BASIS_FILE = Path("state/cost_basis.json")

# Below this base-asset amount a position counts as closed (basis cleared).
_CLOSED_DUST = 1e-9


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def read_basis() -> Dict[str, Dict[str, float]]:
    if not _BASIS_FILE.exists():
        return {}
    try:
        data = json.loads(_BASIS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read cost basis: %s", exc)
        return {}


def _write_basis(data: Dict[str, Dict[str, float]]) -> None:
    try:
        _BASIS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _BASIS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _BASIS_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write cost basis: %s", exc)


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------

def apply_orders(results: List[OrderResult]) -> None:
    """Fold the fills from a completed rebalance into the running cost basis.

    Only live, successful orders with a known fill price are counted; dry-run and
    failed orders carry price=None and are ignored. Fills for the same base across
    multiple client accounts are aggregated into the one portfolio-level entry.
    """
    data = read_basis()
    changed = False
    for r in results:
        if r.status != "ok" or r.price is None or not r.filled:
            continue
        base = r.symbol.split("/")[0]
        entry = data.setdefault(base, {"amount": 0.0, "quote_invested": 0.0})
        amount = float(entry.get("amount") or 0.0)
        invested = float(entry.get("quote_invested") or 0.0)
        filled = float(r.filled)
        cost = float(r.cost if r.cost is not None else r.price * filled)

        if r.side == "buy":
            entry["amount"] = amount + filled
            entry["quote_invested"] = invested + cost
        else:  # sell — reduce amount/invested proportionally; avg cost unchanged
            new_amount = max(0.0, amount - filled)
            entry["amount"] = new_amount
            entry["quote_invested"] = invested * (new_amount / amount) if amount > 0 else 0.0
        changed = True

    if changed:
        _prune(data)
        _write_basis(data)


def reconcile(snapshot: Dict[str, Any]) -> None:
    """Clamp the ledger to the live holdings in `snapshot`.

    Drops bases no longer held (position closed → basis reset) and scales a base's
    amount/invested down to reality when fees/dust left less than the ledger thinks,
    keeping the average entry price fixed. Never invents basis for new holdings
    (that's seed_missing's job).
    """
    data = read_basis()
    if not data:
        return
    held = _held_amounts(snapshot)
    changed = False
    for base in list(data.keys()):
        live = held.get(base, 0.0)
        ledger_amount = float(data[base].get("amount") or 0.0)
        if live <= _CLOSED_DUST:
            del data[base]
            changed = True
        elif live < ledger_amount and ledger_amount > 0:
            avg = float(data[base].get("quote_invested") or 0.0) / ledger_amount
            data[base]["amount"] = live
            data[base]["quote_invested"] = live * avg
            changed = True
    if changed:
        _write_basis(data)


def seed_missing(snapshot: Dict[str, Any]) -> None:
    """Adopt any held base that has no ledger entry, at its current price.

    Covers positions that predate this feature and externally deposited coins:
    avg_entry = current price → 0% profit, so it only alerts after real growth.
    (A future enhancement could reconstruct true basis via ccxt fetch_my_trades.)
    """
    data = read_basis()
    changed = False
    for p in snapshot.get("positions") or []:
        base = p.get("base")
        amount = float(p.get("amount") or 0.0)
        value = float(p.get("value") or 0.0)
        if not base or amount <= _CLOSED_DUST or value <= 0.0:
            continue
        if base not in data:
            data[base] = {"amount": amount, "quote_invested": value}  # avg_entry = value/amount
            changed = True
    if changed:
        _write_basis(data)


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------

def profit_table(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-position unrealized P&L, sorted by profit % descending.

    Returns rows {base, amount, value, pct, price, avg_entry, profit_pct} for each
    held position. profit_pct is None when there is no usable basis yet.
    """
    data = read_basis()
    rows: List[Dict[str, Any]] = []
    for p in snapshot.get("positions") or []:
        base = p.get("base")
        amount = float(p.get("amount") or 0.0)
        value = float(p.get("value") or 0.0)
        if not base or amount <= _CLOSED_DUST:
            continue
        price = value / amount if amount > 0 else 0.0
        entry = data.get(base) or {}
        led_amount = float(entry.get("amount") or 0.0)
        led_invested = float(entry.get("quote_invested") or 0.0)
        avg_entry = (led_invested / led_amount) if led_amount > 0 else None
        profit_pct = ((price - avg_entry) / avg_entry * 100.0) if avg_entry and avg_entry > 0 else None
        rows.append({
            "base": base,
            "amount": amount,
            "value": value,
            "pct": p.get("pct"),
            "price": price,
            "avg_entry": avg_entry,
            "profit_pct": profit_pct,
        })
    rows.sort(key=lambda r: (r["profit_pct"] is None, -(r["profit_pct"] or 0.0)))
    return rows


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _held_amounts(snapshot: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in snapshot.get("positions") or []:
        base = p.get("base")
        if base:
            out[base] = float(p.get("amount") or 0.0)
    return out


def _prune(data: Dict[str, Dict[str, float]]) -> None:
    """Drop fully-sold entries so the file doesn't accumulate zeroed positions."""
    for base in [b for b, e in data.items() if float(e.get("amount") or 0.0) <= _CLOSED_DUST]:
        del data[base]
