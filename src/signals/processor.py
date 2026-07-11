from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..config import AppConfig
from ..models import AssetAllocation, SystemState, TargetPortfolio, WebhookSignal

_ALLOC_PRECISION = 4
_STATE_FILE = Path("state/initial_state.json")

logger = logging.getLogger(__name__)


@dataclass
class _ExecItem:
    """What goes on the execution queue: the portfolio to execute plus the
    owned-symbol updates to apply to each signalling system on success."""
    portfolio: TargetPortfolio
    system_updates: Dict[str, Set[str]] = field(default_factory=dict)


class SignalProcessor:
    """Receives target-allocation signals, batches them within a time window,
    and executes per-system rebalances that only touch each system's own assets.

    Key design:
    - Each signal system owns a slice of the portfolio (tracked via owned_symbols).
    - When system_X signals, only its assets are sold/bought.  Other systems'
      assets are listed in TargetPortfolio.protected_symbols and left untouched.
    - owned_symbols is updated only after a rebalance succeeds, so a failed
      rebalance leaves the state consistent with what is actually on the exchange.
    - On a fresh deploy with no state file, seed state/initial_state.json so the
      processor knows which system owns which assets before the first signal.

    Two asyncio tasks keep collection and execution independent:
        _batch_loop  — collects signals, aggregates into _ExecItem, puts on queue.
        _exec_loop   — drains the queue one item at a time.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._signal_queue: asyncio.Queue[WebhookSignal] = asyncio.Queue()
        self._exec_queue:   asyncio.Queue[_ExecItem]    = asyncio.Queue()
        self._window_s: float = config.execution.aggregation_window_ms / 1000.0
        self._min_allocation: float = config.execution.min_allocation
        self._batch_task: asyncio.Task | None = None
        self._exec_task:  asyncio.Task | None = None
        # system_name → SystemState (last signal + assets this system owns)
        self._system_states: Dict[str, SystemState] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, on_target) -> None:
        self._on_target = on_target
        self._load_state()
        self._batch_task = asyncio.create_task(self._batch_loop(), name="signal-batch")
        self._exec_task  = asyncio.create_task(self._exec_loop(),  name="signal-exec")
        logger.info(
            "Signal processor started (window=%d ms)", self._config.execution.aggregation_window_ms
        )

    async def stop(self) -> None:
        for task in (self._batch_task, self._exec_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(self, signal: WebhookSignal) -> None:
        if self._config.signals.get(signal.system) is None:
            logger.warning("Unknown signal system '%s' – dropping signal", signal.system)
            return
        assets = "  ".join(f"{a.symbol} {a.allocation*100:.1f}%" for a in signal.allocations) or "CASH"
        logger.debug("Queued: system=%s  [%s]", signal.system, assets)
        await self._signal_queue.put(signal)

    # ------------------------------------------------------------------
    # Internal — batch collection
    # ------------------------------------------------------------------

    async def _batch_loop(self) -> None:
        while True:
            try:
                first = await self._signal_queue.get()
                batch: List[WebhookSignal] = [first]
                logger.debug("Batch opened by %s, quiescence window = %.0f ms",
                             first.system, self._window_s * 1000)

                while True:
                    try:
                        sig = await asyncio.wait_for(
                            self._signal_queue.get(), timeout=self._window_s
                        )
                        batch.append(sig)
                        logger.debug("Added %s to batch (%d signals so far)", sig.system, len(batch))
                    except asyncio.TimeoutError:
                        break

                logger.info("Processing batch of %d signal(s)", len(batch))
                item = self._aggregate(batch)

                if item is None:
                    logger.info("Batch produced no executable rebalance")
                else:
                    await self._exec_queue.put(item)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Unexpected error in batch loop: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Internal — sequential execution
    # ------------------------------------------------------------------

    async def _exec_loop(self) -> None:
        while True:
            try:
                item = await self._exec_queue.get()
                try:
                    results = await self._on_target(item.portfolio)
                    failed = [r for r in results if r.status == "error"]
                    if failed:
                        logger.error(
                            "Order(s) failed — system states NOT updated, "
                            "re-sending the same signal will retry: %s",
                            [(r.symbol, r.error) for r in failed],
                        )
                    else:
                        # Only update owned_symbols when all orders succeeded so that
                        # a failed rebalance leaves the state consistent with the exchange.
                        for sys_name, new_symbols in item.system_updates.items():
                            if sys_name in self._system_states:
                                old = self._system_states[sys_name]
                                self._system_states[sys_name] = SystemState(
                                    signal=old.signal,
                                    owned_symbols=new_symbols,
                                )
                        self._save_state()
                except Exception as exc:
                    logger.error(
                        "Rebalance execution failed — system states NOT updated, "
                        "re-sending the same signal will retry: %s", exc, exc_info=True,
                    )
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, signals: List[WebhookSignal]) -> Optional[_ExecItem]:
        """Per-system aggregation.

        Only the systems that sent a signal in this batch are rebalanced.
        Assets owned by silent systems are listed in protected_symbols so the
        exchange layer never sells them.

        Signal deduplication: if a system's new signal is identical to what is
        already in its state, it is not included in the rebalance.
        """
        signalling_systems: set[str] = set()

        for sig in signals:
            if self._config.signals.get(sig.system) is None:
                continue

            existing = self._system_states.get(sig.system)

            # Deduplicate: skip if this system's signal hasn't changed
            if existing is not None and self._signal_unchanged(existing.signal, sig.allocations):
                logger.info("System %s signal unchanged — skipping", sig.system)
                continue

            # Update signal; owned_symbols stays as-is until rebalance succeeds
            self._system_states[sig.system] = SystemState(
                signal=sig.allocations,
                owned_symbols=existing.owned_symbols if existing else set(),
            )
            signalling_systems.add(sig.system)
            alloc_str = "  ".join(
                f"{a.symbol} {a.allocation*100:.1f}%" for a in sig.allocations
            ) or "CASH"
            logger.info("State updated: system=%s  [%s]", sig.system, alloc_str)

        # Save updated signals even if none are new (idempotent)
        self._save_state()

        if not signalling_systems:
            return None

        # Protection split into two cases:
        #
        # 1. purely_protected — asset is exclusively owned by a silent system
        #    AND neither targeted by nor previously held by any signalling system.
        #    → Skip it entirely in the rebalance (handles drift correctly).
        #
        # 2. protected_fraction — asset is shared: a silent system holds it AND
        #    a signalling system either targets it or previously held it.
        #    → Allow selling/buying only beyond the silent system's expected floor
        #    (floor = silent_alloc × silent_weight, as a fraction of total value).

        silent_owned: Set[str] = set()
        for sys_name, sys_state in self._system_states.items():
            if sys_name not in signalling_systems:
                silent_owned.update(sys_state.owned_symbols)

        signalling_owned: Set[str] = set()
        for sys_name in signalling_systems:
            signalling_owned.update(self._system_states[sys_name].owned_symbols)

        # Build weighted targets from signalling systems only
        weighted: Dict[str, float] = defaultdict(float)
        for sys_name in signalling_systems:
            sys_cfg = self._config.signals[sys_name]
            sys_state = self._system_states[sys_name]
            for asset in sys_state.signal:
                weighted[asset.symbol] += asset.allocation * sys_cfg.weight

        targets = {sym: alloc for sym, alloc in weighted.items() if alloc >= self._min_allocation}

        target_bases = {sym.split("/")[0] for sym in targets}

        # Assets silent systems have that signalling systems are also involved with
        overlap = silent_owned & (target_bases | signalling_owned)

        protected_symbols = silent_owned - target_bases - signalling_owned  # purely protected

        protected_fraction: Dict[str, float] = {}
        for sys_name, sys_state in self._system_states.items():
            if sys_name not in signalling_systems:
                sys_cfg = self._config.signals[sys_name]
                for asset in sys_state.signal:
                    base = asset.symbol.split("/")[0]
                    if base in overlap:
                        protected_fraction[base] = (
                            protected_fraction.get(base, 0.0) + asset.allocation * sys_cfg.weight
                        )

        # Expected allocation for purely protected symbols — for display only.
        protected_targets: Dict[str, float] = {}
        for sys_name, sys_state in self._system_states.items():
            if sys_name not in signalling_systems:
                sys_cfg = self._config.signals[sys_name]
                for asset in sys_state.signal:
                    base = asset.symbol.split("/")[0]
                    if base in protected_symbols:
                        protected_targets[base] = (
                            protected_targets.get(base, 0.0) + asset.allocation * sys_cfg.weight
                        )

        if not targets:
            logger.info("CASH signal from %s — will sell their assets (protected: %s)",
                        signalling_systems, protected_symbols)
            portfolio = TargetPortfolio(
                targets={}, quote="",
                protected_symbols=protected_symbols,
                protected_fraction=protected_fraction,
                protected_targets=protected_targets,
            )
        else:
            quotes = {sym.split("/")[1] for sym in targets if "/" in sym}
            if not quotes:
                logger.error("No symbols with a recognisable quote currency (expected BASE/QUOTE)")
                return None
            if len(quotes) > 1:
                logger.error("Mixed quote currencies in target portfolio: %s — cannot rebalance", quotes)
                return None
            portfolio = TargetPortfolio(
                targets=targets,
                quote=quotes.pop(),
                protected_symbols=protected_symbols,
                protected_fraction=protected_fraction,
                protected_targets=protected_targets,
            )

        logger.info("Target portfolio: %r", portfolio)

        # After a successful rebalance, each signalling system's owned_symbols
        # becomes the base assets in its new signal (or empty for CASH).
        system_updates = {
            sys_name: {a.symbol.split("/")[0] for a in self._system_states[sys_name].signal}
            for sys_name in signalling_systems
        }

        return _ExecItem(portfolio=portfolio, system_updates=system_updates)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                sys_name: {
                    "signal": [a.model_dump() for a in st.signal],
                    "owned_symbols": list(st.owned_symbols),
                }
                for sys_name, st in self._system_states.items()
            }
            tmp = _STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, _STATE_FILE)
            logger.debug("State saved to %s", _STATE_FILE)
        except Exception as exc:
            logger.error("Failed to save state to %s — changes are in-memory only and will be lost on restart: %s", _STATE_FILE, exc)

    def _load_state(self) -> None:
        """Load system states from state/initial_state.json.

        This file is the single source of truth — it is read on startup and
        overwritten after every successful rebalance.

        Format:
        {
          "system_1": {
            "signal": [{"symbol": "BTC/USD", "allocation": 1.0}],
            "owned_symbols": ["BTC"]
          },
          "system_2": {
            "signal": [{"symbol": "ETH/USD", "allocation": 0.6}],
            "owned_symbols": ["ETH"]
          }
        }
        """
        if not _STATE_FILE.exists():
            logger.info(
                "No state/initial_state.json found — starting fresh. "
                "Create it to seed per-system ownership before the first signal arrives."
            )
            return
        try:
            raw = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            loaded = []
            for sys_name, data in (raw.get("system_states") or raw).items():
                if sys_name not in self._config.signals:
                    continue
                self._system_states[sys_name] = SystemState(
                    signal=[AssetAllocation(**a) for a in (data.get("signal") or [])],
                    owned_symbols=set(data.get("owned_symbols") or []),
                )
                loaded.append(sys_name)
            if loaded:
                logger.info("Loaded state from %s — systems: %s", _STATE_FILE, loaded)
        except Exception as exc:
            logger.warning("Could not load state from %s: %s", _STATE_FILE, exc)

    def set_state(self, raw: dict) -> List[str]:
        """Replace in-memory state from a dict and persist it to disk.

        Accepts the same format as initial_state.json. Returns the list of
        system names that were loaded. Unknown system names are ignored.
        """
        # Accept {"systems": {...}}, {"system_states": {...}}, or flat {"system_1": {...}}
        entries = raw.get("systems") or raw.get("system_states") or raw
        loaded = []
        for sys_name, data in entries.items():
            if sys_name not in self._config.signals:
                continue
            self._system_states[sys_name] = SystemState(
                signal=[AssetAllocation(**a) for a in (data.get("signal") or [])],
                owned_symbols=set(data.get("owned_symbols") or []),
            )
            loaded.append(sys_name)
        self._save_state()
        logger.info("State overwritten via API — systems: %s", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def preview_portfolio(self, signals: Optional[List[WebhookSignal]] = None) -> Optional[TargetPortfolio]:
        """Build the TargetPortfolio that would result from these signals without mutating state.

        If signals is None or empty, all systems with a current non-empty signal are treated
        as signalling — useful for simulating the full current state.
        """
        effective_states = dict(self._system_states)
        signalling_systems: Set[str] = set()

        if signals:
            for sig in signals:
                if self._config.signals.get(sig.system) is None:
                    continue
                existing = effective_states.get(sig.system)
                effective_states[sig.system] = SystemState(
                    signal=sig.allocations,
                    owned_symbols=existing.owned_symbols if existing else set(),
                )
                signalling_systems.add(sig.system)
        else:
            signalling_systems = {
                name for name, state in effective_states.items()
                if state.signal and name in self._config.signals
            }

        if not signalling_systems:
            return None

        silent_owned: Set[str] = set()
        for sys_name, sys_state in effective_states.items():
            if sys_name not in signalling_systems:
                silent_owned.update(sys_state.owned_symbols)

        signalling_owned: Set[str] = set()
        for sys_name in signalling_systems:
            signalling_owned.update(effective_states[sys_name].owned_symbols)

        weighted: Dict[str, float] = defaultdict(float)
        for sys_name in signalling_systems:
            sys_cfg = self._config.signals[sys_name]
            for asset in effective_states[sys_name].signal:
                weighted[asset.symbol] += asset.allocation * sys_cfg.weight

        targets = {sym: alloc for sym, alloc in weighted.items() if alloc >= self._min_allocation}
        target_bases = {sym.split("/")[0] for sym in targets}
        overlap = silent_owned & (target_bases | signalling_owned)
        protected_symbols = silent_owned - target_bases - signalling_owned

        protected_fraction: Dict[str, float] = {}
        for sys_name, sys_state in effective_states.items():
            if sys_name not in signalling_systems:
                sys_cfg = self._config.signals[sys_name]
                for asset in sys_state.signal:
                    base = asset.symbol.split("/")[0]
                    if base in overlap:
                        protected_fraction[base] = (
                            protected_fraction.get(base, 0.0) + asset.allocation * sys_cfg.weight
                        )

        protected_targets: Dict[str, float] = {}
        for sys_name, sys_state in effective_states.items():
            if sys_name not in signalling_systems:
                sys_cfg = self._config.signals[sys_name]
                for asset in sys_state.signal:
                    base = asset.symbol.split("/")[0]
                    if base in protected_symbols:
                        protected_targets[base] = (
                            protected_targets.get(base, 0.0) + asset.allocation * sys_cfg.weight
                        )

        if not targets:
            return TargetPortfolio(
                targets={}, quote="",
                protected_symbols=protected_symbols,
                protected_fraction=protected_fraction,
                protected_targets=protected_targets,
            )

        quotes = {sym.split("/")[1] for sym in targets if "/" in sym}
        if not quotes or len(quotes) > 1:
            return None

        return TargetPortfolio(
            targets=targets,
            quote=quotes.pop(),
            protected_symbols=protected_symbols,
            protected_fraction=protected_fraction,
            protected_targets=protected_targets,
        )

    def signal_changed(self, signal: WebhookSignal) -> bool:
        """True if this system's incoming allocation differs from its current state.

        Used by the in-process signal generator to skip execution when the daily
        signal hasn't moved. Because state is only advanced after a *successful*
        rebalance, a failed rebalance leaves the old state in place — so the next
        identical signal still reports changed=True and is retried.
        """
        existing = self._system_states.get(signal.system)
        if existing is None:
            return True
        return not self._signal_unchanged(existing.signal, signal.allocations)

    @staticmethod
    def _signal_unchanged(old: List[AssetAllocation], new: List[AssetAllocation]) -> bool:
        """Return True if both allocation lists represent the same portfolio."""
        if len(old) != len(new):
            return False
        old_map = {a.symbol: round(a.allocation, _ALLOC_PRECISION) for a in old}
        new_map = {a.symbol: round(a.allocation, _ALLOC_PRECISION) for a in new}
        return old_map == new_map
