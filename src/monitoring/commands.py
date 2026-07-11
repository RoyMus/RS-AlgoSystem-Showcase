"""Inbound Telegram command listener.

The fly app is private (no public ingress), so a Telegram webhook can't reach it —
instead we long-poll getUpdates from a daemon thread (same fire-and-forget spirit as
TelegramNotifier's sender). Commands are accepted ONLY from the configured chat_id;
anything else is ignored.

Trades never fire from a single keyword: `rebalance` / `rebalance <SYM>` reply with the
plan and arm a short-lived pending action; a follow-up `yes` actually executes it,
through the same `execute_target` choke point the signal processor uses (so the
cost-basis ledger and position snapshot stay consistent).

Supported commands (case-insensitive):
    status | profit | positions   → current per-position unrealized P&L
    rebalance                      → plan a full re-apply of current strategy targets
    rebalance <SYM>                → plan a trim of <SYM> back to its strategy weight
    yes | confirm                  → execute the armed plan
    cancel | no                    → discard the armed plan
    help                           → command list
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Awaitable, Callable, Dict, List, Optional

import requests

from ..models import OrderResult, TargetPortfolio
from . import cost_basis

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"
_LONG_POLL_S = 50
_HTTP_TIMEOUT_S = _LONG_POLL_S + 10
_PENDING_TTL_S = 300  # an armed plan must be confirmed within 5 minutes


class TelegramCommandListener:
    def __init__(
        self,
        *,
        bot_token: Optional[str],
        chat_id: Optional[str],
        loop: asyncio.AbstractEventLoop,
        notifier,
        processor,
        manager,
        execute_target: Callable[[TargetPortfolio], Awaitable[List[OrderResult]]],
    ) -> None:
        self.enabled = bool(bot_token and chat_id)
        self._token = bot_token
        self._chat_id = str(chat_id) if chat_id else None
        # Optional second factor: numeric Telegram user id(s) allowed to issue
        # commands, read from the TELEGRAM_ALLOWED_USER_IDS env var (fly secret).
        # chat_id alone is not enough if the chat is ever a group; binding to the
        # sender's user id locks trading to specific people. Empty set = disabled.
        self._allowed_user_ids = self._load_allowed_user_ids()
        self._loop = loop
        self._notifier = notifier
        self._processor = processor
        self._manager = manager
        self._execute_target = execute_target
        self._offset = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Armed plan awaiting a `yes`. Mutated only on the event loop (in _handle).
        self._pending: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            logger.info("Telegram command listener disabled — no bot_token/chat_id.")
            return
        self._thread = threading.Thread(target=self._poll_loop, name="telegram-commands", daemon=True)
        self._thread.start()
        logger.info("Telegram command listener active (chat_id=%s).", self._chat_id)
        if self._allowed_user_ids:
            logger.info("Telegram commands restricted to user id(s): %s", sorted(self._allowed_user_ids))
        else:
            logger.warning(
                "TELEGRAM_ALLOWED_USER_IDS not set — commands accepted from ANY sender in "
                "chat_id=%s. Set it to your Telegram user id to lock trading to yourself.",
                self._chat_id,
            )

    @staticmethod
    def _load_allowed_user_ids() -> set[int]:
        """Parse TELEGRAM_ALLOWED_USER_IDS (comma/; separated numeric ids) → set of ints."""
        raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
        ids: set[int] = set()
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                logger.warning("Ignoring non-numeric TELEGRAM_ALLOWED_USER_IDS entry: %r", part)
        return ids

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Poll thread
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        url = _API.format(token=self._token, method="getUpdates")
        # Skip any backlog queued while the bot was offline so we don't replay old
        # commands on boot: fetch once with a short timeout and advance the offset.
        try:
            resp = requests.get(url, params={"timeout": 0, "offset": -1}, timeout=20)
            for upd in resp.json().get("result", []):
                self._offset = max(self._offset, upd["update_id"] + 1)
        except Exception as exc:  # noqa: BLE001
            print(f"[telegram-commands] initial drain failed: {exc}", file=sys.stderr)

        while not self._stop.is_set():
            try:
                resp = requests.get(
                    url,
                    params={"timeout": _LONG_POLL_S, "offset": self._offset},
                    timeout=_HTTP_TIMEOUT_S,
                )
                for upd in resp.json().get("result", []):
                    self._offset = max(self._offset, upd["update_id"] + 1)
                    self._dispatch(upd)
            except Exception as exc:  # noqa: BLE001 — never let the poll thread die
                print(f"[telegram-commands] poll error: {exc}", file=sys.stderr)
                time.sleep(5)

    def _dispatch(self, update: Dict) -> None:
        msg = update.get("message") or update.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        if not text or chat_id != self._chat_id:
            return  # ignore non-text and any chat that isn't the owner
        if self._allowed_user_ids:
            from_id = (msg.get("from") or {}).get("id")
            if from_id not in self._allowed_user_ids:
                logger.warning("Ignoring Telegram command from unauthorized user id=%s", from_id)
                return
        # Hand off to the event loop so command handling can await async work safely.
        asyncio.run_coroutine_threadsafe(self._handle(text), self._loop)

    # ------------------------------------------------------------------
    # Command handling (runs on the event loop)
    # ------------------------------------------------------------------

    async def _handle(self, text: str) -> None:
        try:
            parts = text.split()
            cmd = parts[0].lower()
            arg = parts[1].upper() if len(parts) > 1 else None

            if cmd in ("status", "profit", "positions"):
                await self._reply(await self._profit_text())
            elif cmd == "rebalance" and arg is None:
                await self._plan_full()
            elif cmd == "rebalance":
                await self._plan_trim(arg)
            elif cmd in ("yes", "confirm"):
                await self._confirm()
            elif cmd in ("cancel", "no"):
                self._pending = None
                await self._reply("Cancelled.")
            elif cmd in ("help", "/help", "start", "/start"):
                await self._reply(_HELP)
            else:
                await self._reply("Unknown command.\n\n" + _HELP)
        except Exception as exc:  # noqa: BLE001
            logger.error("Telegram command '%s' failed: %s", text, exc, exc_info=True)
            await self._reply(f"⚠️ Command failed: {exc}")

    async def _plan_full(self) -> None:
        target = self._processor.preview_portfolio()
        if target is None or not target.targets:
            await self._reply("Strategy is fully in cash — nothing to rebalance.")
            self._pending = None
            return
        plan = await self._describe_plan(target)
        self._arm(target, "Full portfolio rebalance to current strategy targets")
        await self._reply(
            "♻️ <b>Full rebalance</b> — re-apply current strategy targets:\n"
            f"{_fmt_targets(target)}\n\n{plan}\n\nReply <b>yes</b> to execute, <b>cancel</b> to abort."
        )

    async def _plan_trim(self, base: str) -> None:
        snapshot = await self._manager.equity_snapshot()
        held = {p["base"]: p for p in (snapshot.get("positions") or []) if (p.get("amount") or 0) > 0}
        if base not in held:
            await self._reply(f"You don't hold {base}. Held: {', '.join(sorted(held)) or '(none)'}.")
            return

        preview = self._processor.preview_portfolio()
        quote = (preview.quote if preview else None) or snapshot.get("quote") or "USDT"
        target_frac = 0.0
        if preview:
            target_frac = next(
                (f for s, f in preview.targets.items() if s.split("/")[0] == base), 0.0
            )

        protected = set(held) - {base}
        targets = {f"{base}/{quote}": target_frac} if target_frac > 0 else {}
        target = TargetPortfolio(targets=targets, quote=quote, protected_symbols=protected)

        plan = await self._describe_plan(target)
        if target_frac > 0:
            desc = f"Trim {base} back to its strategy target ({target_frac*100:.1f}%)"
            head = f"✂️ <b>Trim {base}</b> back to target {target_frac*100:.1f}% (proceeds → cash):"
        else:
            desc = f"Sell {base} fully (strategy no longer targets it)"
            head = f"✂️ <b>Sell {base} fully</b> — strategy target is 0% (proceeds → cash):"
        self._arm(target, desc)
        await self._reply(f"{head}\n\n{plan}\n\nReply <b>yes</b> to execute, <b>cancel</b> to abort.")

    async def _confirm(self) -> None:
        pending = self._pending
        if not pending or pending["expires"] < time.monotonic():
            self._pending = None
            await self._reply("Nothing pending to confirm (or it expired). Send a rebalance command first.")
            return
        self._pending = None
        await self._reply(f"⏳ Executing: {pending['desc']} …")
        results = await self._execute_target(pending["target"])
        await self._reply(_fmt_results(results, pending["desc"]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _arm(self, target: TargetPortfolio, desc: str) -> None:
        self._pending = {"target": target, "desc": desc, "expires": time.monotonic() + _PENDING_TTL_S}

    async def _describe_plan(self, target: TargetPortfolio) -> str:
        """Dry-run the target across all clients and summarise the orders it would place."""
        results = await self._manager.simulate_rebalance(target)
        lines: List[str] = []
        for r in results:
            if r.status == "error":
                lines.append(f"❌ {r.side.upper()} {r.symbol} — {(r.error or '')[:80]}")
            else:
                lines.append(f"• {r.side.upper()} {r.symbol} {r.quantity:.6g} [{r.exchange}]")
        return "<b>Plan:</b>\n" + ("\n".join(lines) if lines else "• (already at target — no trades)")

    async def _profit_text(self) -> str:
        snapshot = await self._manager.equity_snapshot()
        rows = cost_basis.profit_table(snapshot)
        quote = snapshot.get("quote") or ""
        if not rows:
            return "No open positions."
        lines = ["📊 <b>Positions (unrealized P&amp;L)</b>", ""]
        for r in rows:
            pp = r["profit_pct"]
            pp_s = f"{pp:+.1f}%" if pp is not None else "n/a"
            entry_s = f"{r['avg_entry']:.4g}" if r["avg_entry"] else "?"
            lines.append(
                f"• <b>{r['base']}</b> {pp_s}  ({entry_s} → {r['price']:.4g})  "
                f"{r['value']:,.2f} {quote} ({(r['pct'] or 0)*100:.0f}%)"
            )
        return "\n".join(lines)

    async def _reply(self, text: str) -> None:
        # Reuse the notifier's sender (always goes to the configured chat); dedup_key
        # is None so command replies are never throttled.
        self._notifier.notify(text)


_HELP = (
    "<b>Commands</b>\n"
    "• <code>status</code> — positions &amp; unrealized P&amp;L\n"
    "• <code>rebalance</code> — re-apply strategy targets to the whole portfolio\n"
    "• <code>rebalance SOL</code> — trim one position back to its strategy target\n"
    "• <code>yes</code> / <code>cancel</code> — confirm or abort a planned rebalance"
)


def _fmt_targets(target: TargetPortfolio) -> str:
    return "  ".join(f"{s.split('/')[0]} {f*100:.0f}%" for s, f in target.targets.items()) or "CASH"


def _fmt_results(results: List[OrderResult], desc: str) -> str:
    if not results:
        return f"✅ {desc}: nothing to trade (already at target)."
    failed = [r for r in results if r.status == "error"]
    head = "⚠️ <b>Done with errors</b>" if failed else "✅ <b>Done</b>"
    lines = [f"{head} — {desc}", ""]
    for r in results:
        mark = "❌" if r.status == "error" else "✅"
        line = f"{mark} {r.side.upper()} {r.symbol} {r.quantity:.6g} [{r.exchange}]"
        if r.status == "error" and r.error:
            line += f" — {r.error[:100]}"
        lines.append(line)
    return "\n".join(lines)
