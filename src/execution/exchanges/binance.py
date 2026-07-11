from __future__ import annotations

from typing import Any, Dict, Optional

import ccxt.async_support as ccxt

from .base import BaseExchange


class BinanceExchange(BaseExchange):
    """Binance spot exchange implementation."""

    EXCHANGE_ID = "binance"

    # Binance defaultType values: "spot" | "future" (USDT-M) | "delivery" (COIN-M)
    _TYPE_MAP = {"spot": "spot", "futures": "future"}

    def _build_client(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool,
        passphrase: Optional[str],
    ) -> ccxt.Exchange:
        binance_type = self._TYPE_MAP.get(self.market_type, "spot")
        client = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": binance_type,
                },
            }
        )
        if testnet:
            client.set_sandbox_mode(True)
        return client

    async def fetch_balance(self) -> Dict[str, Any]:
        return await self._client.fetch_balance()
