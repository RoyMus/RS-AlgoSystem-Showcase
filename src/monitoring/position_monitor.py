"""Profit monitor — alert on Telegram when a position is up past a threshold.

Periodically values the portfolio, computes each position's unrealized P&L against the
cost-basis ledger, and for any position at/above `profit_threshold_pct` sends a prompt
inviting a rebalance. The prompt is throttled per base (default once / 24h) and the
last-alert timestamps are persisted so a restart doesn't re-spam.

The user acts on the prompt via the inbound command listener (`rebalance <SYM>` / `yes`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from ..config import PositionMonitorConfig
from . import cost_basis
from .notifier import TelegramNotifier

logger = logging.getLogger(__name__)

_ALERT_STATE_FILE = Path("state/profit_alerts.json")


class PositionMonitor:
    def __init__(self, cfg: PositionMonitorConfig, manager, processor, notifier: TelegramNotifier) -> None:
        self._cfg = cfg
        self._manager = manager
        self._processor = processor
        self._notifier = notifier
        self._task: Optional[asyncio.Task] = None
        self._last_alert: Dict[str, float] = self._load_alert_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._cfg.enabled:
            logger.info("Position monitor disabled (position_monitor.enabled=false).")
            return
        if not self._notifier.enabled:
            logger.info("Position monitor disabled — Telegram notifier not configured.")
            return
        self._task = asyncio.create_task(self._loop(), name="position-monitor")
        logger.info(
            "Position monitor started — alert at +%.0f%%, every %d min.",
            self._cfg.profit_threshold_pct, self._cfg.check_interval_minutes,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        interval = max(60, self._cfg.check_interval_minutes * 60)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.check_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # never let the loop die
                logger.error("Position monitor error: %s", exc, exc_info=True)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    async def check_once(self) -> None:
        snapshot = await self._manager.equity_snapshot()
        # Keep the ledger honest before reading it.
        cost_basis.reconcile(snapshot)
        cost_basis.seed_missing(snapshot)
        rows = cost_basis.profit_table(snapshot)

        preview = self._processor.preview_portfolio()
        target_by_base: Dict[str, float] = {}
        if preview:
            for sym, frac in preview.targets.items():
                target_by_base[sym.split("/")[0]] = frac

        quote = snapshot.get("quote") or ""
        now = time.time()
        for r in rows:
            pp = r["profit_pct"]
            if pp is None or pp < self._cfg.profit_threshold_pct:
                continue
            base = r["base"]
            if self._throttled(base, now):
                continue
            self._mark_alerted(base, now)
            self._notifier.notify(self._build_alert(r, target_by_base.get(base), quote))

    def _build_alert(self, row: dict, target_frac: Optional[float], quote: str) -> str:
        base = row["base"]
        entry = f"{row['avg_entry']:.4g}" if row["avg_entry"] else "?"
        cur = f"{row['price']:.4g}"
        cur_pct = (row["pct"] or 0.0) * 100.0
        tgt = f"{target_frac*100:.0f}%" if target_frac is not None else "0% (not in strategy)"
        return (
            f"📈 <b>{base}</b> is up <b>{row['profit_pct']:+.1f}%</b> "
            f"(entry {entry} → {cur}).\n"
            f"Now {row['value']:,.2f} {quote} = {cur_pct:.0f}% of portfolio (strategy target {tgt}).\n\n"
            f"Reply <code>rebalance {base}</code> to trim it back to target, "
            f"or <code>rebalance</code> for the whole portfolio."
        )

    # ------------------------------------------------------------------
    # Throttle state (persisted)
    # ------------------------------------------------------------------

    def _throttled(self, base: str, now: float) -> bool:
        last = self._last_alert.get(base)
        if last is None:
            return False
        return (now - last) < self._cfg.alert_throttle_hours * 3600

    def _mark_alerted(self, base: str, now: float) -> None:
        self._last_alert[base] = now
        self._save_alert_state()

    def _load_alert_state(self) -> Dict[str, float]:
        """Load persisted last-alert times (epoch seconds) so the throttle window
        survives restarts."""
        if not _ALERT_STATE_FILE.exists():
            return {}
        try:
            raw = json.loads(_ALERT_STATE_FILE.read_text(encoding="utf-8"))
            return {b: float(ts) for b, ts in raw.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load profit-alert state: %s", exc)
            return {}

    def _save_alert_state(self) -> None:
        try:
            _ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _ALERT_STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._last_alert, indent=2), encoding="utf-8")
            os.replace(tmp, _ALERT_STATE_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save profit-alert state: %s", exc)
