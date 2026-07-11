"""Telegram notifier + a logging handler that turns ERROR logs into instant alerts.

Design notes:
  - Sends run on a single daemon worker thread draining a queue, so a slow or failing
    Telegram API call never blocks the asyncio event loop (and therefore never delays
    trading) or the logging call site.
  - Duplicate alerts (same dedup key) are suppressed for `throttle_seconds` so a
    recurring error can't flood the chat.
  - The worker reports its own failures to stderr only — never via `logging` — so a
    failed send can't trigger the ERROR-log handler and cause an alert→log→alert loop.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"
_SEND_TIMEOUT_S = 10
# Telegram hard-limits messages to 4096 chars.
_MAX_LEN = 3900


@dataclass
class _Msg:
    text: str
    dedup_key: Optional[str]


class TelegramNotifier:
    """Fire-and-forget Telegram sender with dedup/throttling.

    Call `notify(text, dedup_key=...)` from anywhere (sync or async). When disabled
    (no token/chat id) every call is a no-op, so callers never need to check.
    """

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        throttle_seconds: int = 300,
    ) -> None:
        self.enabled = bool(bot_token and chat_id)
        self._token = bot_token
        self._chat_id = chat_id
        self._throttle_s = max(0, throttle_seconds)
        self._queue: "queue.Queue[_Msg]" = queue.Queue(maxsize=1000)
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        if self.enabled:
            self._worker = threading.Thread(
                target=self._run, name="telegram-notifier", daemon=True
            )
            self._worker.start()
            logger.info("Telegram notifier active (chat_id=%s, throttle=%ds).", chat_id, self._throttle_s)
        else:
            logger.info("Telegram notifier disabled — no bot_token/chat_id configured.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, text: str, *, dedup_key: Optional[str] = None) -> None:
        """Enqueue a message. No-op when disabled or throttled."""
        if not self.enabled:
            return
        if dedup_key is not None and self._throttle_s > 0:
            now = time.monotonic()
            with self._lock:
                last = self._last_sent.get(dedup_key, 0.0)
                if now - last < self._throttle_s:
                    return
                self._last_sent[dedup_key] = now
        try:
            self._queue.put_nowait(_Msg(text=text[:_MAX_LEN], dedup_key=dedup_key))
        except queue.Full:
            print("[telegram-notifier] queue full — dropping message", file=sys.stderr)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        url = _API.format(token=self._token)
        while True:
            msg = self._queue.get()
            try:
                resp = requests.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": msg.text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=_SEND_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    # Never log via `logging` here — would re-enter the ERROR handler.
                    print(
                        f"[telegram-notifier] send failed {resp.status_code}: {resp.text[:200]}",
                        file=sys.stderr,
                    )
            except Exception as exc:  # noqa: BLE001 — must not propagate from worker
                print(f"[telegram-notifier] send error: {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()


class TelegramLogHandler(logging.Handler):
    """Logging handler that forwards ERROR+ records to Telegram as instant alerts.

    Attach to the root logger so it captures every `logger.error(...)` already in the
    codebase — failed orders, signal timeouts, price/symbol-not-found, balance/earn
    errors — without touching any call site.
    """

    def __init__(self, notifier: TelegramNotifier, level: int = logging.ERROR) -> None:
        super().__init__(level=level)
        self._notifier = notifier

    def emit(self, record: logging.LogRecord) -> None:
        # Recursion guard: ignore the notifier's own logger (it only logs lifecycle info,
        # but be safe in case future code logs an error there).
        if record.name == __name__:
            return
        try:
            msg = record.getMessage()
            text = f"⚠️ <b>{record.levelname}</b> [{record.name}]\n{msg}"
            if record.exc_info:
                text += f"\n<pre>{self.format(record)[:1500]}</pre>"
            # Dedup on logger + message prefix so repeated identical errors are throttled.
            self._notifier.notify(text, dedup_key=f"{record.name}:{msg[:80]}")
        except Exception:  # noqa: BLE001 — a logging handler must never raise
            pass
