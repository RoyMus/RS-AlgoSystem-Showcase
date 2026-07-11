from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import ccxt.async_support as ccxt

from .base import BaseExchange

logger = logging.getLogger(__name__)


class BybitExchange(BaseExchange):
    """Bybit spot exchange implementation."""

    EXCHANGE_ID = "bybit"

    # Bybit defaultType values: "spot" | "linear" (USDT perps) | "inverse" (coin-margined)
    _TYPE_MAP = {"spot": "spot", "futures": "linear"}

    def _build_client(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool,
        passphrase: Optional[str],
    ) -> ccxt.Exchange:
        bybit_type = self._TYPE_MAP.get(self.market_type, "spot")
        options: Dict[str, Any] = {
            "defaultType": bybit_type,
            "adjustForTimeDifference": True,   # auto-corrects clock skew vs Bybit server
            "recvWindow": 20000,               # 20s window — covers clock skew before calibration kicks in
            "createMarketBuyOrderRequiresPrice": False,  # futures market buys don't need a price
            "slippage": 0.005,                 # 0.5% — keeps IOC limit price within Bybit's markPrice×1.05 ceiling
        }
        client = ccxt.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": options,
            }
        )
        if self.demo:
            # Demo trading uses a separate endpoint (api-demo.bybit.com). It must be
            # switched on via enable_demo_trading() — which rewrites the hostname —
            # not by setting the demoTrading option, which leaves the host on mainnet.
            # set_sandbox_mode() instead points at the public testnet (a different env).
            client.enable_demo_trading(True)
        elif testnet:
            client.set_sandbox_mode(True)
        return client

    def _map_symbol(self, symbol: str) -> str:
        """For futures, convert BASE/QUOTE → BASE/QUOTE:QUOTE (ccxt linear perpetual format)."""
        if self.market_type == "futures" and ":" not in symbol and "/" in symbol:
            quote = symbol.split("/")[1]
            return f"{symbol}:{quote}"
        return symbol

    async def _fetch_long_positions(self, target) -> Dict[str, Any]:  # type: ignore[override]
        """Bybit linear: pass settleCoin so all USDT-margined positions are returned."""
        try:
            positions = await self._client.fetch_positions(
                None, params={"settleCoin": self.quote_currency}
            )
        except Exception:
            # Fallback: fetch by target symbols only
            try:
                mapped = [self._map_symbol(s) for s in target.targets] or None
                positions = await self._client.fetch_positions(mapped)
            except Exception as exc:
                logger.error(
                    "[%s] Failed to fetch positions: %s", self.name, exc
                )
                return {}

        held: Dict[str, float] = {}
        for pos in positions:
            contracts = float(pos.get("contracts") or 0)
            if pos.get("side") == "long" and contracts > 0:
                raw_sym = pos.get("symbol", "")
                base = raw_sym.split("/")[0] if "/" in raw_sym else ""
                if base:
                    held[base] = contracts
        return held

    async def fetch_balance(self) -> Dict[str, Any]:
        bybit_type = self._TYPE_MAP.get(self.market_type, "spot")
        return await self._client.fetch_balance({"type": bybit_type})
