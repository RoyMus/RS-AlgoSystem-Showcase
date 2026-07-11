"""In-process RS Algo signal generator.

Runs one or more strategy instances (RsDynamicOg.compute_signal) on a daily
schedule, entirely inside the executor process. Each instance maps to a signal
`system` and can have its own asset basket (config overrides). Computed target
allocations are fed straight to the SignalProcessor — no external webhook, so
the trade-triggering path never crosses the network.

Robustness:
  - Each (blocking, network-bound) computation runs in a worker thread with a
    hard timeout, so a hung data fetch can never wedge the server's event loop.
  - Execution only fires when an instance's allocation actually changes vs. what
    that system last sent (see SignalProcessor.signal_changed), avoiding churn.
  - Instances run sequentially (gentle on the data APIs); when more than one
    changes, both enqueue inside the processor's batch window and rebalance
    together with proper cross-system netting/protection.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..config import GeneratorSystemConfig, SignalGeneratorConfig
from ..models import WebhookSignal
from ..monitoring import strategy_equity
from .processor import SignalProcessor

logger = logging.getLogger(__name__)


class SignalScheduler:
    def __init__(self, cfg: SignalGeneratorConfig, processor: SignalProcessor) -> None:
        self._cfg = cfg
        self._processor = processor
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._cfg.enabled:
            logger.info("Signal generator disabled — not scheduling.")
            return
        if not self._cfg.systems:
            logger.warning("Signal generator enabled but no systems configured — nothing to run.")
            return

        known = self._processor._config.signals
        unknown = [s.system for s in self._cfg.systems if s.system not in known]
        if unknown:
            logger.error(
                "Signal generator references system(s) %s not defined under signals: in "
                "config.yaml — refusing to start.", unknown,
            )
            return

        self._task = asyncio.create_task(self._loop(), name="signal-generator")
        logger.info(
            "Signal generator scheduled daily at %s UTC (systems=%s, timeout=%ds).",
            self._cfg.run_at_utc, [s.system for s in self._cfg.systems], self._cfg.timeout_seconds,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Schedule loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        if self._cfg.run_on_start:
            await self.run_once()
        while True:
            try:
                delay = self._seconds_until_next_run()
                logger.info("Next signal generation in %.1f h.", delay / 3600.0)
                await asyncio.sleep(delay)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # never let the loop die
                logger.error("Signal scheduler error: %s", exc, exc_info=True)
                await asyncio.sleep(60)

    def _seconds_until_next_run(self) -> float:
        hh, mm = (int(x) for x in self._cfg.run_at_utc.split(":"))
        now = datetime.now(timezone.utc)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def run_once(self) -> List[dict]:
        """Compute every configured instance and enqueue the ones that changed.
        Returns a list of per-instance result dicts (failed/timed-out ones omitted)."""
        results: List[dict] = []
        for spec in self._cfg.systems:
            r = await self._run_one(spec)
            if r is not None:
                results.append(r)
        return results

    async def _run_one(self, spec: GeneratorSystemConfig) -> Optional[dict]:
        logger.info("Generating signal for %s …", spec.system)
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._compute, spec.overrides),
                timeout=self._cfg.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Signal generation for %s timed out after %ds — skipping.",
                spec.system, self._cfg.timeout_seconds,
            )
            return None
        except Exception as exc:
            logger.error("Signal generation for %s failed — skipping: %s", spec.system, exc, exc_info=True)
            return None

        result["system"] = spec.system
        if result.get("strategy_curve"):
            strategy_equity.write_curve(result["strategy_curve"])
        alloc_str = result["webhook"]
        signal = WebhookSignal(system=spec.system, allocations=alloc_str)

        if not self._processor.signal_changed(signal):
            logger.info("%s signal for %s unchanged (%s) — no execution.",
                        spec.system, result["date"], alloc_str)
            return result

        logger.info("%s signal CHANGED for %s → %s — enqueuing.",
                    spec.system, result["date"], alloc_str)
        await self._processor.enqueue(signal)
        return result

    @staticmethod
    def _compute(overrides: dict) -> dict:
        # Heavy import (numpy/pandas/yfinance) — only pulled in when actually computing.
        from . import engine
        cfg = {**engine.CONFIG, **overrides}
        return engine.compute_signal(cfg)
