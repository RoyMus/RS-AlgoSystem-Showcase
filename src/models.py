from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, field_validator


class AssetAllocation(BaseModel):
    """A single asset + its target allocation fraction within a signal."""

    symbol: str       # e.g. "ETH/USDT", normalised from "ETH-USDT"
    allocation: float # 0.0 < x <= 1.0

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper().replace("-", "/")

    @field_validator("allocation")
    @classmethod
    def valid_allocation(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("allocation must be between 0 (exclusive) and 1.0 (inclusive)")
        return v


CASH_SIGNAL = "CASH"  # Special keyword: sell all holdings, move to quote currency


def _parse_allocations_string(raw: str) -> List[dict]:
    """Parse TradingView-style allocation string.

    Accepted formats (case-insensitive, '-' treated as '/'):
        "ETH/USDT 22.5% | SOL/USDT 22.5% | PAXG/USDT 50%"
        "BTC-USD 100%"
        "CASH"  ← special: sell everything and hold quote currency
    """
    # Accept bare "CASH" or "CASH 100%" — percentage is ignored
    if re.fullmatch(r"CASH(\s+[\d.]+\s*%?)?", raw.strip(), re.IGNORECASE):
        return []  # empty allocations = sell all holdings

    result = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        # Strip optional TradingView exchange prefix (e.g. "CRYPTO:", "BINANCE:")
        part = re.sub(r"^[A-Za-z]+:", "", part)
        m = re.fullmatch(r"([A-Za-z0-9/_\-]+)\s+([\d.]+)\s*%?", part)
        if not m:
            raise ValueError(
                f"Cannot parse '{part}' — expected 'SYMBOL PCT%'  e.g. 'ETH/USDT 22.5%'"
            )
        symbol, pct = m.group(1), float(m.group(2))
        result.append({"symbol": symbol, "allocation": pct / 100.0})
    if not result:
        raise ValueError("allocations string is empty — use 'CASH' to liquidate all holdings")
    return result


class WebhookSignal(BaseModel):
    """Target portfolio allocation from one signal system.

    There is no buy/sell side — the allocations describe what fraction of
    the portfolio should be in each asset after rebalancing.

    `allocations` accepts either:
      - A pipe-separated string:  "ETH/USDT 22.5% | SOL/USDT 22.5% | PAXG/USDT 50%"
      - A list of objects:        [{"symbol": "ETH/USDT", "allocation": 0.225}, ...]
    """

    system: str
    allocations: List[AssetAllocation]

    @field_validator("allocations", mode="before")
    @classmethod
    def parse_allocations(cls, v: Union[str, list]) -> list:
        if isinstance(v, str):
            return _parse_allocations_string(v)
        return v


class SystemState(BaseModel):
    """Per-signal-system state: the last signal it sent and the base assets it currently owns."""

    signal: List[AssetAllocation]
    owned_symbols: Set[str] = set()  # base currencies held by this system, e.g. {"ZEC", "ETH"}


class TargetPortfolio(BaseModel):
    """Target allocation for the systems that signalled in a given batch.

    `targets` maps each symbol to the fraction of *total* portfolio value that
    should be allocated to it after rebalancing.
    `quote` is the common quote currency inferred from the symbols (e.g. "USDT").
    `protected_symbols` lists base currencies owned by *other* systems that must
    not be sold, even if they are not in `targets`.
    """

    targets: Dict[str, float]       # symbol → allocation fraction of total portfolio
    quote: str
    protected_symbols: Set[str] = set()
    # Base currencies owned exclusively by silent systems — never sell these at all.
    protected_fraction: Dict[str, float] = {}
    # Base currencies shared between systems — sell/buy only beyond this floor
    # (floor = silent system's expected allocation × weight, as a fraction of total).
    protected_targets: Dict[str, float] = {}
    # Expected allocation for purely protected symbols (silent system signal × weight).
    # Used only for display — does not affect execution.

    def __repr__(self) -> str:
        parts = "  ".join(f"{sym} {frac:.1%}" for sym, frac in self.targets.items())
        protected = f"  protected={self.protected_symbols}" if self.protected_symbols else ""
        return f"TargetPortfolio({parts}  quote={self.quote}{protected})"


class OrderResult(BaseModel):
    """Result of a single order placed (or simulated) on an exchange."""

    exchange: str
    symbol: str
    side: str         # "buy" | "sell"
    allocation: float
    quantity: float   # base-asset amount
    order_id: Optional[str] = None
    status: str       # "ok" | "dry_run" | "error"
    error: Optional[str] = None
    # Actual fill details, populated for live "ok" orders so the cost-basis ledger
    # can track true average entry. None for dry_run/error (ledger ignores those).
    price: Optional[float] = None    # average fill price in quote currency
    cost: Optional[float] = None     # total quote spent/received (price × filled)
    filled: Optional[float] = None   # base-asset amount actually filled
