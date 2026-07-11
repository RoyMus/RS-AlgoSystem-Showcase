from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Type

from ..config import AppConfig
from .exchanges import BaseExchange, BinanceExchange, BybitExchange, KrakenExchange
from ..models import OrderResult, TargetPortfolio

logger = logging.getLogger(__name__)

_POSITIONS_FILE = Path("state/positions.json")

_EXCHANGE_REGISTRY: Dict[str, Type[BaseExchange]] = {
    "bybit": BybitExchange,
    "binance": BinanceExchange,
    "kraken": KrakenExchange,
}


class ExchangeManager:
    """Manages all configured exchange clients and fans out rebalance orders to every account."""

    def __init__(self, config: AppConfig) -> None:
        self._clients: List[BaseExchange] = []
        self._init_clients(config)

    def _init_clients(self, config: AppConfig) -> None:
        for exchange_name, exc_cfg in config.enabled_exchanges.items():
            exchange_class = _EXCHANGE_REGISTRY.get(exchange_name)
            if exchange_class is None:
                logger.warning("Unknown exchange '%s' in config – skipping", exchange_name)
                continue

            for client_cfg in exc_cfg.clients:
                label = client_cfg.label or exchange_name
                try:
                    # Per-client exposure overrides global; fall back to execution default
                    exposure = (
                        client_cfg.portfolio_exposure
                        if client_cfg.portfolio_exposure is not None
                        else config.execution.portfolio_exposure
                    )
                    client = exchange_class(
                        api_key=client_cfg.api_key,
                        api_secret=client_cfg.api_secret,
                        testnet=client_cfg.testnet,
                        passphrase=client_cfg.passphrase,
                        label=label,
                        dry_run=client_cfg.dry_run,
                        demo=client_cfg.demo,
                        quote_currency=exc_cfg.quote_currency,
                        min_trade_value=config.execution.min_trade_value,
                        order_delay_ms=config.execution.order_delay_ms,
                        portfolio_exposure=exposure,
                        market_type=client_cfg.market_type,
                        leverage=client_cfg.leverage,
                        margin_mode=client_cfg.margin_mode,
                        use_earn=client_cfg.use_earn,
                    )
                    mode = "dry_run" if client_cfg.dry_run else ("testnet" if client_cfg.testnet else "live")
                    logger.info("Initialized client: %s (%s)", label, mode)
                    self._clients.append(client)
                except Exception as exc:
                    logger.error("Failed to initialize client '%s': %s", label, exc)

    async def execute_rebalance(self, target: TargetPortfolio) -> List[OrderResult]:
        """Rebalance every client toward the target portfolio concurrently.

        Returns a flat list of OrderResult (one per order × client).
        """
        if not self._clients:
            return []

        logger.info(
            "Rebalancing %d client(s) toward: %r", len(self._clients), target
        )

        results_per_client = await asyncio.gather(
            *[client.rebalance(target) for client in self._clients],
            return_exceptions=True,
        )

        flat: List[OrderResult] = []
        for client, result in zip(self._clients, results_per_client):
            if isinstance(result, Exception):
                logger.error("Rebalance failed for client '%s': %s", client.name, result)
            else:
                flat.extend(result)

        return flat

    async def snapshot_positions(self) -> None:
        """Fetch real balances from every client and write them to state/positions.json.

        Called on startup (to bootstrap from actual exchange state) and after each
        successful rebalance so the file always reflects what is really held.
        """
        clients_data: Dict[str, Any] = {}
        for client in self._clients:
            try:
                balance = await client.fetch_balance()
                spot: Dict[str, float] = {}
                for currency, bal in balance.items():
                    if not isinstance(bal, dict):
                        continue
                    total = float(bal.get("total") or 0.0)
                    if total > 0:
                        spot[currency] = total
                earn = await client.fetch_earn_balance()
                clients_data[client.name] = {"spot": spot, "earn": earn}
            except Exception as exc:
                logger.warning("Could not snapshot positions for %s: %s", client.name, exc)

        data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "clients": clients_data,
        }
        try:
            _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _POSITIONS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, _POSITIONS_FILE)
            logger.info("Positions snapshot saved → %s", _POSITIONS_FILE)
        except Exception as exc:
            logger.warning("Failed to save positions snapshot: %s", exc)

    async def simulate_rebalance(self, target: TargetPortfolio) -> List[OrderResult]:
        """Dry-run rebalance on every client: fetches real balances but places no orders."""
        if not self._clients:
            return []
        results_per_client = await asyncio.gather(
            *[client.simulate_rebalance(target) for client in self._clients],
            return_exceptions=True,
        )
        flat: List[OrderResult] = []
        for client, result in zip(self._clients, results_per_client):
            if isinstance(result, Exception):
                logger.error("Simulate failed for client '%s': %s", client.name, result)
            else:
                flat.extend(result)
        return flat

    async def equity_snapshot(self) -> Dict[str, Any]:
        """Value every client and aggregate into one portfolio snapshot.

        Returns:
          {
            "timestamp": ISO-8601 UTC,
            "quote": common quote currency,
            "total_value": float,            # spot + holdings + earn across all clients
            "quote_free": float,
            "earn_value": float,
            "positions": [{base, amount, value, pct}],  # merged across clients
            "clients": { name: <per-client compute_equity dict> },
          }
        Used by the equity sampler, the dashboard and the weekly report.
        """
        per_client = await asyncio.gather(
            *[c.compute_equity() for c in self._clients],
            return_exceptions=True,
        )

        clients_data: Dict[str, Any] = {}
        total_value = 0.0
        quote_free = 0.0
        earn_value = 0.0
        quote = ""
        merged: Dict[str, Dict[str, float]] = {}

        for client, result in zip(self._clients, per_client):
            if isinstance(result, Exception):
                logger.error("Equity snapshot failed for client '%s': %s", client.name, result)
                continue
            result["exposure"] = getattr(client, "_portfolio_exposure", 1.0)
            clients_data[client.name] = result
            total_value += result["total_value"]
            quote_free += result["quote_free"]
            earn_value += result["earn_value"]
            quote = quote or result["quote"]
            for p in result["positions"]:
                agg = merged.setdefault(p["base"], {"amount": 0.0, "value": 0.0})
                agg["amount"] += p["amount"]
                agg["value"] += p["value"]

        positions = [
            {
                "base": base,
                "amount": agg["amount"],
                "value": agg["value"],
                "pct": (agg["value"] / total_value) if total_value > 0 else 0.0,
            }
            for base, agg in merged.items()
        ]
        positions.sort(key=lambda p: -p["value"])

        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "quote": quote,
            "total_value": total_value,
            "quote_free": quote_free,
            "earn_value": earn_value,
            "positions": positions,
            "clients": clients_data,
        }

    async def fetch_cashflows(self, since_ms: int) -> Dict[str, float]:
        """Net external cashflow per client (deposits − withdrawals, quote currency) since
        since_ms. Failures map to 0.0 so a flaky deposit-history call never blocks sampling."""
        results = await asyncio.gather(
            *[c.fetch_cashflows(since_ms) for c in self._clients],
            return_exceptions=True,
        )
        out: Dict[str, float] = {}
        for client, r in zip(self._clients, results):
            if isinstance(r, Exception):
                logger.warning("Cashflow fetch failed for client '%s': %s", client.name, r)
                out[client.name] = 0.0
            else:
                out[client.name] = float(r)
        return out

    async def close(self) -> None:
        await asyncio.gather(*[c.close() for c in self._clients])

    @property
    def client_labels(self) -> List[str]:
        return [c.name for c in self._clients]
