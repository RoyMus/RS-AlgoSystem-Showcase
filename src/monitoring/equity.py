"""Periodic portfolio valuation → equity time-series.

Every `equity_sample_interval_minutes` the sampler values the whole portfolio
(spot + holdings + earn, via ExchangeManager.equity_snapshot) and:
  - appends one compact JSON line to state/equity_history.jsonl  (the equity curve)
  - overwrites state/equity_latest.json                          (fast dashboard read)

Both files live on the existing `state_data` fly mount, so the curve survives restarts.
The append-only history keeps writes cheap and crash-safe.

P&L note: pnl() reports the raw equity delta over a window. Deposits/withdrawals are
NOT adjusted for — fine for a single funded account; revisit if flows become frequent.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HISTORY_FILE = Path("state/equity_history.jsonl")
_LATEST_FILE = Path("state/equity_latest.json")


def _dominant_asset_of(positions: List[Dict[str, Any]]) -> str:
    """Largest-by-value holding in a position list → used to colour an account's
    curve segment. Returns CASH when there are no (non-dust) token positions."""
    if not positions:
        return "CASH"
    top = max(positions, key=lambda p: p.get("value") or 0.0)
    if (top.get("value") or 0.0) <= 0.0:
        return "CASH"
    return top.get("base") or "CASH"


class EquitySampler:
    def __init__(self, manager, interval_minutes: int = 60) -> None:
        self._manager = manager
        self._interval_s = max(60, int(interval_minutes) * 60)
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle (mirrors SignalScheduler)
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="equity-sampler")
        logger.info("Equity sampler started (interval=%.0f min).", self._interval_s / 60)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        # Sample once on startup so the dashboard/report have data immediately.
        await self.sample_once()
        while True:
            try:
                await asyncio.sleep(self._interval_s)
                await self.sample_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # never let the loop die
                logger.error("Equity sampler error: %s", exc, exc_info=True)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    async def sample_once(self) -> Optional[Dict[str, Any]]:
        snapshot = await self._manager.equity_snapshot()
        # Skip empty snapshots: when every client fails to value (transient exchange/
        # credential error) the snapshot has no clients and total_value 0. Recording it
        # writes a 0-equity point that corrupts the curve and shows a false -100% drawdown.
        # Drop it instead so the dashboard keeps the last good sample.
        if not snapshot.get("clients"):
            logger.warning(
                "Equity sample skipped — no account could be valued "
                "(transient exchange/credential error); keeping last good sample."
            )
            return snapshot
        # Net external cashflow (deposits − withdrawals) per account since the previous
        # sample, so the curve can divide funding out of performance (TWR). Keyed off the
        # last recorded sample's timestamp; first-ever sample has no prior point → no flow.
        flows: Dict[str, float] = {}
        prev = _last_record()
        prev_ts = _parse_ts(prev.get("ts")) if prev else None
        if prev_ts is not None:
            since_ms = int(prev_ts.timestamp() * 1000)
            try:
                flows = await self._manager.fetch_cashflows(since_ms)
            except Exception as exc:
                logger.warning("Could not fetch cashflows for equity sample: %s", exc)

        # Compact record for the time-series. We keep the merged positions plus a
        # light per-account value breakdown (value/free/earn only — not the verbose
        # per-client positions, which stay in equity_latest.json) so the dashboard
        # can track each account's actual P&L over time when multiple accounts run
        # the same strategy.
        record = {
            "ts": snapshot["timestamp"],
            "total_value": round(snapshot["total_value"], 4),
            "flow": round(sum(flows.values()), 4),  # portfolio-wide net cashflow this interval
            "quote": snapshot["quote"],
            "positions": [
                {"base": p["base"], "value": round(p["value"], 4), "pct": round(p["pct"], 4)}
                for p in snapshot["positions"]
            ],
            "accounts": {
                name: {
                    "value": round(c.get("total_value", 0.0), 4),
                    "free": round(c.get("quote_free", 0.0), 4),
                    "earn": round(c.get("earn_value", 0.0), 4),
                    "exposure": round(c.get("exposure", 1.0), 4),
                    "flow": round(flows.get(name, 0.0), 4),  # net cashflow this interval

                    # This account's OWN largest holding (or CASH) so the dashboard
                    # colours each account's curve by what *it* held — accounts with
                    # different exposures/positions no longer all show the portfolio's
                    # dominant asset.
                    "asset": _dominant_asset_of(c.get("positions") or []),
                }
                for name, c in (snapshot.get("clients") or {}).items()
            },
        }
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _HISTORY_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            tmp = _LATEST_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            os.replace(tmp, _LATEST_FILE)
            logger.info(
                "Equity sample: %.2f %s recorded → %s",
                snapshot["total_value"], snapshot["quote"], _HISTORY_FILE,
            )
        except Exception as exc:
            logger.error("Failed to record equity sample: %s", exc)
        return snapshot


# ----------------------------------------------------------------------
# Read helpers (used by the dashboard routes and the weekly reporter)
# ----------------------------------------------------------------------

def read_history(since: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
    """Return equity history records, optionally only those at/after `since` (UTC)."""
    if not _HISTORY_FILE.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in _HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                ts = _parse_ts(rec.get("ts"))
                if ts is None or ts < since:
                    continue
            out.append(rec)
    except Exception as exc:
        logger.warning("Could not read equity history: %s", exc)
    return out


def _last_record() -> Optional[Dict[str, Any]]:
    """The most recent equity-history record, or None. Cheap tail read for the sampler."""
    hist = read_history()
    return hist[-1] if hist else None


def twr_adjust(values: List[float], flows: List[float]) -> List[float]:
    """Deposit-adjusted equity (time-weighted return), same dollar units, anchored at the
    first sample. ``flows[i]`` is the net deposit (− withdrawal) during the interval ending
    at sample i (flows[0] is ignored). Each interval's pure return (value minus that
    interval's cashflow, over the prior value) is chained, so deposits/withdrawals don't
    register as gains — fixing curves that otherwise jump by the full funded amount."""
    if not values:
        return []
    out = [values[0]]
    for i in range(1, len(values)):
        prev = values[i - 1]
        flow = flows[i] if i < len(flows) else 0.0
        r = ((values[i] - flow) - prev) / prev if prev > 0 else 0.0
        out.append(out[-1] * (1.0 + r))
    return out


def read_latest() -> Optional[Dict[str, Any]]:
    """Return the most recent full equity snapshot, or None if none recorded yet."""
    if not _LATEST_FILE.exists():
        return None
    try:
        return json.loads(_LATEST_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read latest equity snapshot: %s", exc)
        return None


def reset_history() -> int:
    """Delete the equity-history file to re-baseline the curve.

    Returns the number of data points removed. Used after a measurement change
    (e.g. enabling Kraken earn) created a discontinuity that poisons the P&L.
    The next sample re-seeds history at the current (correct) equity.
    """
    removed = 0
    if _HISTORY_FILE.exists():
        try:
            removed = sum(1 for ln in _HISTORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip())
            _HISTORY_FILE.unlink()
            logger.info("Equity history reset — removed %d point(s).", removed)
        except Exception as exc:
            logger.warning("Could not reset equity history: %s", exc)
    return removed


def pnl(period: datetime.timedelta) -> Optional[Dict[str, Any]]:
    """Equity P&L over `period`: picks the sample closest to (now − period) as the start.

    Returns {start_value, end_value, abs, pct, start_ts, end_ts} or None if no history.
    """
    history = read_history()
    if not history:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - period

    return _pnl_from(
        [r.get("ts") for r in history],
        [float(r.get("total_value") or 0.0) for r in history],
        [float(r.get("flow") or 0.0) for r in history],
        cutoff,
    )


def pnl_account(name: str, period: datetime.timedelta) -> Optional[Dict[str, Any]]:
    """Per-account P&L over `period`, deposit-adjusted (TWR). The per-account value
    recorded in history is that client's total_value, which already includes earn —
    so this P&L reflects earn. Returns None if the account has no samples."""
    history = read_history()
    ts_list, values, flows = [], [], []
    for r in history:
        acc = (r.get("accounts") or {}).get(name)
        if not acc:
            continue
        ts_list.append(r.get("ts"))
        values.append(float(acc.get("value") or 0.0))
        flows.append(float(acc.get("flow") or 0.0))
    if not values:
        return None
    cutoff = datetime.datetime.now(datetime.timezone.utc) - period
    return _pnl_from(ts_list, values, flows, cutoff)


def _pnl_from(ts_list, values, flows, cutoff) -> Optional[Dict[str, Any]]:
    """TWR P&L from parallel ts/value/flow lists: start = last sample at/before cutoff."""
    if not values:
        return None
    adj = twr_adjust(values, flows)  # deposit-adjusted so funding isn't counted as P&L
    end_i = len(values) - 1
    start_i = None
    for i, ts in enumerate(ts_list):
        t = _parse_ts(ts)
        if t is None:
            continue
        if t <= cutoff:
            start_i = i
        else:
            break
    if start_i is None:
        start_i = 0
    sv, ev = adj[start_i], adj[end_i]
    return {
        "start_value": sv,
        "end_value": ev,
        "abs": ev - sv,
        "pct": ((ev - sv) / sv) if sv > 0 else 0.0,
        "start_ts": ts_list[start_i],
        "end_ts": ts_list[end_i],
    }


def _parse_ts(value: Any) -> Optional[datetime.datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except ValueError:
        return None


if __name__ == "__main__":
    # Deposit of 1000 in the final interval while the market is flat: pure return is
    # +10% (1000→1100), and the deposit must NOT show up as a gain.
    out = twr_adjust([1000.0, 1100.0, 2100.0], [0.0, 0.0, 1000.0])
    assert abs(out[-1] - 1100.0) < 1e-6, out          # deposit divided out
    # No flows → identical to raw compounding of the values.
    assert abs(twr_adjust([100.0, 110.0], [0.0, 0.0])[-1] - 110.0) < 1e-6
    # Withdrawal (negative flow) is likewise removed.
    out_w = twr_adjust([1000.0, 900.0], [0.0, -100.0])
    assert abs(out_w[-1] - 1000.0) < 1e-6, out_w
    print("equity.twr_adjust self-check OK")
