"""Weekly performance report pushed to Telegram.

Sleeps until the configured weekday + time (UTC), then sends a snapshot of open
positions, total equity, and P&L (7-day + since-inception) drawn from the equity
time-series recorded by EquitySampler.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Optional

from ..config import ReportingConfig
from . import equity
from .notifier import TelegramNotifier

logger = logging.getLogger(__name__)

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# A position worth less than this (quote currency) is treated as dust, not an open
# position — so the portfolio counts as "all cash" once everything is liquidated.
_CASH_DUST = 1.0
# 7-day equity change smaller than this fraction counts as "no real change"
# (e.g. cash idling in earn at ~4% APY moves <0.1% in a week, so it won't trigger a report).
_FLAT_PNL_PCT = 0.001


class WeeklyReporter:
    def __init__(self, cfg: ReportingConfig, manager, notifier: TelegramNotifier) -> None:
        self._cfg = cfg
        self._manager = manager
        self._notifier = notifier
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._notifier.enabled:
            logger.info("Weekly reporter disabled — Telegram notifier not configured.")
            return
        self._task = asyncio.create_task(self._loop(), name="weekly-reporter")
        logger.info(
            "Weekly reporter scheduled %s at %s UTC.",
            self._cfg.weekly_report_day, self._cfg.weekly_report_at_utc,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                delay = self._seconds_until_next_run()
                logger.info("Next weekly report in %.1f h.", delay / 3600.0)
                await asyncio.sleep(delay)
                await self.send_report(force=False)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # never let the loop die
                logger.error("Weekly reporter error: %s", exc, exc_info=True)
                await asyncio.sleep(3600)

    def _seconds_until_next_run(self) -> float:
        hh, mm = (int(x) for x in self._cfg.weekly_report_at_utc.split(":"))
        target_dow = _WEEKDAYS.index(self._cfg.weekly_report_day)
        now = datetime.datetime.now(datetime.timezone.utc)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        days_ahead = (target_dow - now.weekday()) % 7
        target += datetime.timedelta(days=days_ahead)
        if target <= now:
            target += datetime.timedelta(days=7)
        return (target - now).total_seconds()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    async def send_report(self, force: bool = True) -> str:
        """Build and (unless suppressed) send the report. Returns the message text.

        When `force` is False (the scheduled run), the report is suppressed if the
        portfolio is fully in cash AND equity is essentially flat over the week —
        there's nothing worth reporting when the money is just parked. `/report/now`
        passes force=True so a manual check always sends.
        """
        # Take a fresh valuation so the report reflects the moment it's sent, and so the
        # curve has a data point even between scheduled samples.
        snapshot = await self._manager.equity_snapshot()
        text = self._build_message(snapshot)

        if not force and self._is_quiet(snapshot):
            logger.info("Weekly report skipped — all cash and no meaningful change this week.")
            return text

        self._notifier.notify(text)
        logger.info("Weekly report sent.")
        return text

    @staticmethod
    def _is_quiet(snapshot: dict) -> bool:
        """True if the portfolio is fully in cash and 7-day equity change is negligible."""
        has_position = any(
            float(p.get("value") or 0.0) >= _CASH_DUST for p in (snapshot.get("positions") or [])
        )
        if has_position:
            return False
        wk = equity.pnl(datetime.timedelta(days=7))
        return wk is None or abs(wk["pct"]) < _FLAT_PNL_PCT

    def _build_message(self, snapshot: dict) -> str:
        clients = snapshot.get("clients") or {}
        # Which accounts to report: configured list (in order) or every account sampled.
        wanted = self._cfg.report_accounts or list(clients.keys())
        names = [n for n in wanted if n in clients]

        lines = ["📊 <b>Weekly performance report</b>"]
        if not names:
            lines.append("")
            lines.append("<i>No accounts to report.</i>")
            return "\n".join(lines)

        for name in names:
            lines.append("")
            lines.append(self._account_section(name, clients[name]))

        lines.append("")
        lines.append("<i>P&amp;L is time-weighted (deposits/withdrawals divided out) and includes earn.</i>")
        return "\n".join(lines)

    def _account_section(self, name: str, cli: dict) -> str:
        """One per-account block: equity, TWR P&L (incl. earn), and open positions."""
        quote = cli.get("quote") or ""
        total = float(cli.get("total_value") or 0.0)
        earn = float(cli.get("earn_value") or 0.0)
        cash = float(cli.get("quote_free") or 0.0)
        positions = cli.get("positions") or []

        lines = [f"<b>━ {name}</b>"]
        lines.append(f"<b>Equity:</b> {total:,.2f} {quote}")

        wk = equity.pnl_account(name, datetime.timedelta(days=7))
        if wk:
            lines.append(f"<b>7-day P&amp;L:</b> {_fmt_signed(wk['abs'])} {quote}  ({_fmt_pct(wk['pct'])})")
        inc = equity.pnl_account(name, datetime.timedelta(days=3650))
        if inc:
            lines.append(f"<b>Since inception:</b> {_fmt_signed(inc['abs'])} {quote}  ({_fmt_pct(inc['pct'])})")

        lines.append("<b>Open positions:</b>")
        shown = [p for p in positions if p.get("value", 0.0) >= 0.01]
        if shown:
            for p in shown:
                lines.append(f"  • {p['base']}: {p['value']:,.2f} {quote}  ({p.get('pct', 0.0) * 100:.1f}%)")
        else:
            lines.append("  • (none)")
        if cash >= 0.01:
            cash_pct = (cash / total * 100) if total > 0 else 0.0
            lines.append(f"  • Cash ({quote}): {cash:,.2f}  ({cash_pct:.1f}%)")
        if earn >= 0.01:
            earn_pct = (earn / total * 100) if total > 0 else 0.0
            lines.append(f"  • Earn ({quote}): {earn:,.2f}  ({earn_pct:.1f}%)")
        return "\n".join(lines)


def _fmt_signed(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{value:,.2f}"


def _fmt_pct(frac: float) -> str:
    return f"{'+' if frac >= 0 else ''}{frac * 100:.2f}%"
