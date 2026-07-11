from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt

from ...models import OrderResult, TargetPortfolio
from .base import BaseExchange

logger = logging.getLogger(__name__)

_EARN_POLL_INTERVAL_S = 3.0
_EARN_TIMEOUT_S = 120.0
_EARN_MIN_AMOUNT = 0.5   # don't bother depositing/withdrawing less than this
_EARN_RESERVE = 50.0     # leave this much in earn to satisfy Kraken's minimum balance


class KrakenExchange(BaseExchange):
    """Kraken exchange implementation.

    Kraken uses different symbol naming (e.g. XBT/USD instead of BTC/USD).
    The _map_symbol method handles the most common remappings.

    When use_earn=True, idle quote currency is kept in Kraken's flexible Earn
    product between rebalances. Before each rebalance only the minimum amount
    needed to cover net buys is withdrawn; if spot cash already covers the buys
    (e.g. after sells, or a sell-only rebalance) no withdrawal happens at all.
    After the rebalance any remaining free cash is deposited back.
    """

    EXCHANGE_ID = "kraken"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._earn_strategy_id: str | None = None
        self._rebalance_earn_bal: float = 0.0  # injected into fetch_balance during planning

    # ------------------------------------------------------------------
    # BaseExchange overrides
    # ------------------------------------------------------------------

    def _build_client(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool,
        passphrase: Optional[str],
    ) -> ccxt.Exchange:
        # Kraken does not have a public testnet; sandbox mode is a no-op
        return ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    # Kraken suffixes that indicate non-tradeable earn/staking balances
    _EARN_SUFFIXES = (".S", ".M", ".B", ".X", ".F")

    async def fetch_balance(self) -> Dict[str, Any]:
        balance = await self._client.fetch_balance()
        # Strip non-tradeable staked/earn asset entries (e.g. ATOM.S, USD.M, BNB.B)
        # so the rebalancer doesn't try to fetch prices for them.
        balance = {
            k: v for k, v in balance.items()
            if not any(k.endswith(sfx) for sfx in self._EARN_SUFFIXES)
        }
        if self._rebalance_earn_bal > 0:
            # During planning, earn funds are injected here so total_value is computed
            # correctly against the full portfolio. Cleared before live order execution.
            quote_bal = balance.get(self.quote_currency)
            free = float(quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0
            updated = dict(quote_bal) if isinstance(quote_bal, dict) else {}
            updated["free"] = free + self._rebalance_earn_bal
            balance = dict(balance)
            balance[self.quote_currency] = updated
        return balance

    async def rebalance(self, target: TargetPortfolio) -> List[OrderResult]:
        if not self._use_earn:
            return await super().rebalance(target)

        # 1. Resolve any in-flight deallocation from a previous run before planning,
        #    so earn_bal accurately reflects what is settled and withdrawable.
        if not self.dry_run and await self._is_earn_deallocation_pending():
            logger.info("[%s] Earn: deallocation already pending, waiting for it", self.name)
            await self._wait_earn_deallocate()
            await self._wait_earn_settle_in_spot()

        # 2. Fetch earn balance and inject it into fetch_balance() so _plan_rebalance
        #    sizes positions against the full portfolio value (spot + earn + holdings).
        earn_bal = await self._fetch_earn_balance()
        self._rebalance_earn_bal = earn_bal

        plan = await self._plan_rebalance(target)

        # Clear injection before execution — live orders must use the real spot balance.
        self._rebalance_earn_bal = 0.0

        # 3. Compute the minimum earn withdrawal needed to fund the planned buys.
        #    Sells free up cash before buys run, so only the net deficit must come from earn.
        sell_proceeds = sum(amt * plan.prices.get(sym, 0.0) for sym, amt in plan.sells)
        total_buy_cost = sum(cost for _, cost, _ in plan.buys)
        # plan.quote_free included earn via injection; subtract to get actual spot cash.
        actual_spot_free = plan.quote_free - earn_bal
        net_earn_needed = max(0.0, total_buy_cost - sell_proceeds - actual_spot_free)

        earn_available = max(earn_bal - _EARN_RESERVE, 0.0)
        withdraw_amt = min(net_earn_needed, earn_available)

        # 4. Withdraw only what is needed (or nothing if spot covers the buys).
        if withdraw_amt > _EARN_MIN_AMOUNT:
            if self.dry_run:
                logger.info(
                    "[%s] DRY RUN: would withdraw %.4f %s from earn"
                    " (%.4f needed for buys, %.4f available, keeping %.2f reserve)",
                    self.name, withdraw_amt, self.quote_currency,
                    net_earn_needed, earn_available, _EARN_RESERVE,
                )
                self._rebalance_earn_bal = withdraw_amt
            else:
                logger.info(
                    "[%s] Earn: withdrawing %.4f %s (%.4f needed for buys, keeping %.2f reserve)",
                    self.name, withdraw_amt, self.quote_currency, net_earn_needed, _EARN_RESERVE,
                )
                if await self._earn_deallocate(withdraw_amt):
                    await self._wait_earn_deallocate()
                    await self._wait_earn_settle_in_spot(withdraw_amt)
        else:
            logger.info(
                "[%s] Earn: spot cash sufficient for planned buys (%.4f available after sells,"
                " %.4f needed) — no withdrawal",
                self.name, actual_spot_free + sell_proceeds, total_buy_cost,
            )

        # 5. Execute the pre-computed plan.
        try:
            results = await self._execute_rebalance(plan)
        finally:
            self._rebalance_earn_bal = 0.0

        # 6. Deposit any remaining free quote back into earn.
        if not self.dry_run:
            await self._earn_deposit_free()

        return results

    # ------------------------------------------------------------------
    # Earn helpers
    # ------------------------------------------------------------------

    async def _get_earn_strategy_id(self) -> str | None:
        """Discover and cache the flexible earn strategy ID for quote_currency."""
        if self._earn_strategy_id:
            return self._earn_strategy_id
        try:
            resp = await self._client.privatePostEarnStrategies({
                "asset": self.quote_currency,
            })
            for strategy in (resp.get("result") or {}).get("items") or []:
                if (strategy.get("lock_type") or {}).get("type") == "flex":
                    self._earn_strategy_id = strategy["id"]
                    apy = float(
                        (strategy.get("apr_estimate") or {}).get("low") or 0
                    ) * 100
                    logger.info(
                        "[%s] Earn strategy for %s: %s (APY ~%.2f%%)",
                        self.name, self.quote_currency, self._earn_strategy_id, apy,
                    )
                    return self._earn_strategy_id
            logger.warning(
                "[%s] No flexible earn strategy found for %s",
                self.name, self.quote_currency,
            )
        except Exception as exc:
            logger.error("[%s] Failed to fetch earn strategies: %s", self.name, exc)
        return None

    async def _fetch_earn_balance(self) -> float:
        """Return the settled (withdrawable) amount of quote_currency in earn.

        Also caches the strategy_id from the actual allocation so deallocation
        targets the correct strategy rather than the first flex strategy found.
        """
        try:
            resp = await self._client.privatePostEarnAllocations({
                "converted_asset": self.quote_currency,
            })
            for item in (resp.get("result") or {}).get("items") or []:
                if item.get("native_asset") == self.quote_currency:
                    # Use the strategy_id from the allocation, not from the strategies list
                    strategy_id = item.get("strategy_id")
                    if strategy_id:
                        self._earn_strategy_id = strategy_id
                    alloc = item.get("amount_allocated") or {}
                    total   = float((alloc.get("total")      or {}).get("native") or 0.0)
                    exit_q  = float((alloc.get("exit_queue") or {}).get("native") or 0.0)
                    bonding = float((alloc.get("bonding")    or {}).get("native") or 0.0)
                    settled = max(total - exit_q - bonding, 0.0)
                    logger.info(
                        "[%s] Earn balance: strategy=%s total=%.4f exit_queue=%.4f"
                        " bonding=%.4f settled=%.4f",
                        self.name, strategy_id, total, exit_q, bonding, settled,
                    )
                    return settled
        except Exception as exc:
            logger.warning("[%s] Could not fetch earn balance: %s", self.name, exc)
        return 0.0

    async def _earn_deallocate(self, amount: float) -> bool:
        """Initiate a withdrawal from earn. Returns True if the request was accepted."""
        strategy_id = await self._get_earn_strategy_id()
        if not strategy_id:
            return False
        request_amount = math.floor(amount * 1e8) / 1e8
        try:
            await self._client.privatePostEarnDeallocate({
                "strategy_id": strategy_id,
                "amount": f"{request_amount:.8f}",
            })
            logger.info(
                "[%s] Earn: deallocation of %.8f %s requested",
                self.name, request_amount, self.quote_currency,
            )
            return True
        except Exception as exc:
            logger.error("[%s] Earn deallocation failed: %s", self.name, exc)
            return False

    async def _is_earn_deallocation_pending(self) -> bool:
        """Return True if Kraken already has a deallocation in flight for this strategy."""
        strategy_id = await self._get_earn_strategy_id()
        if not strategy_id:
            return False
        try:
            resp = await self._client.privatePostEarnDeallocateStatus({
                "strategy_id": strategy_id,
            })
            return bool((resp.get("result") or {}).get("pending", False))
        except Exception as exc:
            logger.warning("[%s] Could not check earn deallocation status: %s", self.name, exc)
            return False

    async def _wait_earn_deallocate(self) -> None:
        """Poll DeallocateStatus until the withdrawal clears or we time out."""
        strategy_id = await self._get_earn_strategy_id()
        if not strategy_id:
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _EARN_TIMEOUT_S
        while loop.time() < deadline:
            try:
                resp = await self._client.privatePostEarnDeallocateStatus({
                    "strategy_id": strategy_id,
                })
                if not (resp.get("result") or {}).get("pending", True):
                    logger.info("[%s] Earn: withdrawal confirmed", self.name)
                    return
                logger.debug("[%s] Earn: withdrawal still pending…", self.name)
            except Exception as exc:
                logger.warning("[%s] Error polling earn status: %s", self.name, exc)
            await asyncio.sleep(_EARN_POLL_INTERVAL_S)
        logger.warning(
            "[%s] Earn: withdrawal did not confirm within %.0fs — proceeding anyway",
            self.name, _EARN_TIMEOUT_S,
        )

    async def _wait_earn_settle_in_spot(self, expected_increase: float = 0.0) -> None:
        """Poll fetch_balance() until the spot quote balance reflects the earn withdrawal.

        Kraken marks a deallocation as not-pending before the freed funds always
        appear in the spot balance.  Without this second gate the rebalancer reads
        the old (lower) quote balance and sizes positions against a total_value
        that is missing the withdrawn amount — producing allocations that are
        smaller than the requested percentage of actual equity.
        """
        if expected_increase <= 0:
            await asyncio.sleep(_EARN_POLL_INTERVAL_S)
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _EARN_TIMEOUT_S
        try:
            before_bal = await self.fetch_balance()
            baseline = float((before_bal.get(self.quote_currency) or {}).get("free") or 0.0)
        except Exception:
            baseline = 0.0
        target_free = baseline + expected_increase * 0.99  # allow 1% tolerance
        while loop.time() < deadline:
            await asyncio.sleep(_EARN_POLL_INTERVAL_S)
            try:
                bal = await self.fetch_balance()
                free = float((bal.get(self.quote_currency) or {}).get("free") or 0.0)
                if free >= target_free:
                    logger.info(
                        "[%s] Earn: %.4f %s settled in spot balance",
                        self.name, expected_increase, self.quote_currency,
                    )
                    return
                logger.debug(
                    "[%s] Earn: waiting for spot balance to settle (%.2f / %.2f)",
                    self.name, free, target_free,
                )
            except Exception as exc:
                logger.warning("[%s] Error checking spot balance settle: %s", self.name, exc)
        logger.warning(
            "[%s] Earn: spot balance did not reflect withdrawal within %.0fs — proceeding anyway",
            self.name, _EARN_TIMEOUT_S,
        )

    async def fetch_earn_balance(self) -> float:
        """Return the amount of quote_currency currently in flexible earn (0 if earn disabled)."""
        if not self._use_earn:
            return 0.0
        return await self._fetch_earn_balance()

    async def _earn_deposit_free(self) -> None:
        """Deposit all free quote currency back into earn after a rebalance."""
        strategy_id = await self._get_earn_strategy_id()
        if not strategy_id:
            return
        try:
            balance = await self.fetch_balance()
            quote_bal = balance.get(self.quote_currency)
            free = float(quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0
            if free < _EARN_MIN_AMOUNT:
                logger.debug(
                    "[%s] Earn: %.4f %s free — below minimum, skipping deposit",
                    self.name, free, self.quote_currency,
                )
                return
            await self._client.privatePostEarnAllocate({
                "strategy_id": strategy_id,
                "amount": f"{math.floor(free * 1e8) / 1e8:.8f}",
            })
            logger.info("[%s] Earn: deposited %.4f %s", self.name, free, self.quote_currency)
        except Exception as exc:
            logger.warning("[%s] Earn deposit failed: %s", self.name, exc)
