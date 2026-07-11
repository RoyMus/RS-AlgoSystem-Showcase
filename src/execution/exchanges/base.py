from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import ccxt.async_support as ccxt

from ...models import OrderResult, TargetPortfolio

logger = logging.getLogger(__name__)


def _extract_fill(resp: dict, fallback_price: float, fallback_amount: float) -> Tuple[float, float, float]:
    """Pull (price, cost, filled) from a ccxt order response.

    Market orders usually report `average` (volume-weighted fill price), `filled`
    (base amount) and `cost` (quote spent/received). Venues that omit them fall
    back to the price/amount we computed at order time so the cost-basis ledger
    always gets a usable entry. Any missing piece is derived from the others.
    """
    price = resp.get("average") or resp.get("price") or fallback_price or 0.0
    filled = resp.get("filled") or fallback_amount or 0.0
    cost = resp.get("cost")
    if not cost:
        cost = float(price) * float(filled)
    return float(price), float(cost), float(filled)


def _err_mentions(exc: Exception, *needles: str) -> bool:
    """True if the exception text contains any of the given (case-insensitive) needles.

    Used to distinguish a benign "already at the requested value" exchange response
    (e.g. Bybit retCode 110043 "leverage not modified") from a genuine failure.
    """
    text = str(exc).lower()
    return any(n.lower() in text for n in needles)


class _RebalancePlan(NamedTuple):
    target: TargetPortfolio
    quote: str
    quote_free: float
    total_value: float
    held: Dict[str, float]
    prices: Dict[str, float]
    sells: List[Tuple[str, float]]          # (symbol, base_amount)
    buys: List[Tuple[str, float, float]]    # (symbol, quote_cost, target_frac)


class BaseExchange(ABC):
    """Abstract base class for exchange implementations.

    rebalance() implements a delta rebalance:
      1. Fetch all current balances + prices for held and target assets.
      2. Calculate total portfolio value in the quote currency.
      3. For each currently held asset compute the delta vs. target:
           delta > 0  → sell the excess
           delta < 0  → buy the shortfall
         Assets not in the target at all are sold in full.
      4. Execute all sells concurrently, then re-fetch the quote balance.
      5. Execute all buys concurrently from the available quote balance.
    """

    EXCHANGE_ID: str = ""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        passphrase: Optional[str] = None,
        label: Optional[str] = None,
        dry_run: bool = False,
        demo: bool = False,
        quote_currency: str = "USDT",
        min_trade_value: float = 5.0,
        order_delay_ms: int = 1000,
        portfolio_exposure: float = 1.0,
        market_type: str = "spot",
        leverage: int = 1,
        margin_mode: str = "isolated",
        use_earn: bool = False,
    ) -> None:
        self.name = label or self.EXCHANGE_ID
        self.dry_run = dry_run
        self.demo = demo
        self.quote_currency = quote_currency
        self.min_trade_value = min_trade_value
        self._order_delay_s: float = order_delay_ms / 1000.0
        self._portfolio_exposure: float = portfolio_exposure
        self.market_type: str = market_type
        self._leverage: int = leverage
        self._margin_mode: str = margin_mode
        # Track symbols where position params have already been set this session
        self._use_earn = use_earn
        self._leverage_set: Set[str] = set()
        self._client = self._build_client(api_key, api_secret, testnet, passphrase)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_client(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool,
        passphrase: Optional[str],
    ) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.EXCHANGE_ID)
        options: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        if passphrase:
            options["password"] = passphrase
        client: ccxt.Exchange = exchange_class(options)
        if testnet:
            client.set_sandbox_mode(True)
        return client

    def _map_symbol(self, symbol: str) -> str:
        """Override in subclasses to remap symbols to exchange-specific conventions."""
        return symbol

    def _remap_target(self, target: TargetPortfolio) -> TargetPortfolio:
        """Replace the quote currency in every symbol with this exchange's quote_currency.

        Allows a single signal (e.g. ETH/USDT 60%) to be executed on Kraken as ETH/USD
        without changing the signal format. Also handles the CASH (empty targets) case.
        """
        if not target.targets:
            # CASH signal — no symbols to remap, just set this exchange's quote currency
            return TargetPortfolio(
                targets={}, quote=self.quote_currency,
                protected_symbols=target.protected_symbols,
                protected_fraction=target.protected_fraction,
            )
        if target.quote == self.quote_currency:
            return target
        remapped = {
            f"{sym.split('/')[0]}/{self.quote_currency}": frac
            for sym, frac in target.targets.items()
        }
        logger.debug(
            "[%s] Remapped quote %s → %s: %s",
            self.name, target.quote, self.quote_currency,
            {k: f"{v:.1%}" for k, v in remapped.items()},
        )
        return TargetPortfolio(
            targets=remapped, quote=self.quote_currency,
            protected_symbols=target.protected_symbols,
            protected_fraction=target.protected_fraction,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rebalance(self, target: TargetPortfolio) -> List[OrderResult]:
        """Rebalance the account toward the target portfolio.

        Returns one OrderResult per order placed (or simulated in dry_run).
        """
        plan = await self._plan_rebalance(target)
        return await self._execute_rebalance(plan)

    # ------------------------------------------------------------------
    # Plan / execute split (overridable by subclasses)
    # ------------------------------------------------------------------

    async def _plan_rebalance(self, target: TargetPortfolio) -> _RebalancePlan:
        """Compute the rebalance plan: fetch balances/prices and calculate sells/buys.

        Does not execute any orders. Subclasses can override rebalance() and call
        this directly to inspect the plan before executing.
        """
        target = self._remap_target(target)

        if self._portfolio_exposure < 1.0 and target.targets:
            target = TargetPortfolio(
                targets={sym: frac * self._portfolio_exposure for sym, frac in target.targets.items()},
                quote=target.quote,
                protected_symbols=target.protected_symbols,
                protected_fraction=target.protected_fraction,
                protected_targets=target.protected_targets,
            )
            logger.info(
                "[%s] Portfolio exposure: %.0f%% — targets scaled down",
                self.name, self._portfolio_exposure * 100,
            )

        quote = target.quote

        # Defense-in-depth tripwire: a long-only rebalance must never target >100%
        # of equity (that would imply leverage). The signal layer already guarantees
        # fractions sum to ≤1.0; the 1% slack only absorbs per-leg rounding. If this
        # ever fires it is an upstream bug — refuse to trade rather than lever up.
        total_frac = sum(target.targets.values())
        if total_frac > 1.01:
            logger.error(
                "[%s] Target allocation sums to %.1f%% (>100%%) — refusing to rebalance "
                "to avoid a leveraged position: %r",
                self.name, total_frac * 100, target.targets,
            )
            return _RebalancePlan(target, quote, 0.0, 0.0, {}, {}, [], [])

        try:
            balance = await self.fetch_balance()
        except Exception as exc:
            logger.error("[%s] Failed to fetch balance: %s", self.name, exc)
            return _RebalancePlan(target, quote, 0.0, 0.0, {}, {}, [], [])

        quote_bal = balance.get(quote)
        quote_free = float(quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0

        if self.market_type == "futures":
            held = await self._fetch_long_positions(target)
        else:
            held = {}
            for currency, bal in balance.items():
                if currency == quote:
                    continue
                if not isinstance(bal, dict):
                    continue
                free = float(bal.get("free") or 0.0)
                if free > 0:
                    held[currency] = free

        symbols_needed: Set[str] = set(target.targets.keys())
        for base in held:
            symbols_needed.add(f"{base}/{quote}")

        prices = await self._fetch_prices(symbols_needed)

        total_value = quote_free
        for base, amount in held.items():
            sym = f"{base}/{quote}"
            price = prices.get(sym, 0.0)
            if price <= 0:
                logger.warning(
                    "[%s] No price for held asset %s — excluded from total portfolio value; "
                    "target allocations will be inaccurate until price is available",
                    self.name, sym,
                )
            else:
                total_value += amount * price
        logger.info(
            "[%s] Portfolio value: %.4f %s  (quote free: %.4f)",
            self.name, total_value, quote, quote_free,
        )

        sells: List[Tuple[str, float]] = []
        buys:  List[Tuple[str, float, float]] = []

        for base, current_amount in held.items():
            sym = f"{base}/{quote}"
            price = prices.get(sym, 0.0)
            if price <= 0:
                logger.warning("[%s] No price for %s — skipping", self.name, sym)
                continue

            current_value = current_amount * price

            if base in target.protected_symbols:
                continue

            floor_value = total_value * target.protected_fraction.get(base, 0.0)
            available_value = max(0.0, current_value - floor_value)

            target_frac  = target.targets.get(sym, 0.0)
            target_value = total_value * target_frac
            delta_value  = available_value - target_value

            if delta_value > 1e-8:
                sells.append((sym, delta_value / price))
            elif delta_value < -1e-8:
                buys.append((sym, -delta_value, target_frac))

        for sym, frac in target.targets.items():
            base = sym.split("/")[0]
            if base not in held:
                buys.append((sym, total_value * frac, frac))

        sells = [(sym, amt) for sym, amt in sells
                 if amt * prices.get(sym, 0.0) >= self.min_trade_value]
        buys  = [(sym, cost, frac) for sym, cost, frac in buys
                 if cost >= self.min_trade_value]

        self._log_rebalance(held, quote_free, total_value, prices, quote, target, sells, buys)

        return _RebalancePlan(target, quote, quote_free, total_value, held, prices, sells, buys)

    async def _execute_rebalance(self, plan: _RebalancePlan) -> List[OrderResult]:
        """Execute a pre-computed rebalance plan.

        In dry_run mode: logs and returns simulated OrderResults without placing orders.
        In live mode: executes sells, re-fetches balance, scales and executes buys.
        """
        results: List[OrderResult] = []
        quote = plan.quote

        if self.dry_run:
            for sym, amount in plan.sells:
                results.append(OrderResult(
                    exchange=self.name, symbol=sym, side="sell",
                    allocation=0.0, quantity=amount, status="dry_run",
                ))
            for sym, cost, frac in plan.buys:
                price = plan.prices.get(sym, 0.0)
                results.append(OrderResult(
                    exchange=self.name, symbol=sym, side="buy",
                    allocation=frac, quantity=cost / price if price else 0.0,
                    status="dry_run",
                ))
            return results

        # Fetch fresh balance at execution time — important when earn balance was
        # injected during planning (KrakenExchange) but cleared before execution.
        try:
            balance = await self.fetch_balance()
            quote_bal = balance.get(quote)
            quote_free = float(quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0
        except Exception as exc:
            logger.error("[%s] Failed to fetch balance before execution: %s", self.name, exc)
            return results

        # Execute sells sequentially with a delay between each order.
        # Kraken requires a strictly increasing nonce; sending requests too
        # quickly produces colliding timestamps and an "Invalid nonce" error.
        for i, (sym, amount) in enumerate(plan.sells):
            if i > 0:
                await asyncio.sleep(self._order_delay_s)
            result = await self._market_sell(sym, amount)
            results.append(result)

        if plan.sells:
            await asyncio.sleep(self._order_delay_s)
            try:
                balance = await self.fetch_balance()
                quote_bal = balance.get(quote)
                quote_free = float(quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0
            except Exception as exc:
                logger.error("[%s] Failed to fetch balance after sells: %s", self.name, exc)
                return results

        # Scale buys to available quote (in case fees reduced it)
        total_buy_cost = sum(cost for _, cost, _ in plan.buys)
        # Apply a 0.5% fee buffer so trading fees from sells don't cause
        # "Insufficient funds" on the subsequent buys.
        scale = min(0.995, quote_free / total_buy_cost) if total_buy_cost > 0 else 0.0

        if scale > 0:
            for i, (sym, cost, frac) in enumerate(plan.buys):
                if i > 0:
                    await asyncio.sleep(self._order_delay_s)
                result = await self._market_buy(sym, cost * scale, frac)
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_prices(self, symbols: Set[str]) -> Dict[str, float]:
        async def _one(sym: str) -> Tuple[str, float]:
            try:
                ticker = await self._client.fetch_ticker(self._map_symbol(sym))
                return sym, float(ticker.get("last") or ticker.get("bid") or 0.0)
            except Exception as exc:
                logger.warning("[%s] Could not fetch price for %s: %s", self.name, sym, exc)
                return sym, 0.0

        pairs = await asyncio.gather(*[_one(sym) for sym in symbols])
        return dict(pairs)

    async def _fetch_long_positions(self, target: TargetPortfolio) -> Dict[str, float]:
        """For futures accounts: return open long positions as {base_currency: contracts}.

        Only long positions are returned — we never open shorts in a rebalancer.
        Symbols not in the target are also included so excess longs get sold.
        """
        try:
            # Fetch positions for all target symbols + any already open
            mapped = [self._map_symbol(sym) for sym in target.targets]
            positions = await self._client.fetch_positions(mapped or None)
            held: Dict[str, float] = {}
            for pos in positions:
                if (pos.get("side") == "long"
                        and float(pos.get("contracts") or 0) > 0):
                    raw_sym = pos.get("symbol", "")          # e.g. "ETH/USDT:USDT"
                    base = raw_sym.split("/")[0] if "/" in raw_sym else ""
                    if base:
                        held[base] = float(pos["contracts"])
            return held
        except Exception as exc:
            logger.error("[%s] Failed to fetch positions: %s", self.name, exc)
            return {}

    async def _ensure_leverage(self, symbol: str) -> bool:
        """Ensure margin mode + leverage are at the configured values for a futures symbol.

        Returns True only when it is SAFE to OPEN/INCREASE a position — i.e. the
        leverage is confirmed at the configured value (default 1x). If leverage
        cannot be confirmed, returns False so the caller skips the buy rather than
        risk an unintended-leverage position that could be liquidated early.

        Always returns True for spot (leverage is irrelevant there). The result is
        cached per symbol so we only configure once per session.

        Margin mode is best-effort: with leverage pinned to 1x, total notional is
        already ≤ equity (≈ spot exposure), so margin mode is secondary — a failure
        is logged loudly but does not block trading.
        """
        if self.market_type == "spot":
            return True
        if symbol in self._leverage_set:
            return True
        mapped = self._map_symbol(symbol)

        # ── Margin mode (best-effort) ──────────────────────────────────────────
        try:
            await self._client.set_margin_mode(self._margin_mode, mapped)
            logger.info("[%s] Margin mode set to '%s' for %s", self.name, self._margin_mode, symbol)
        except Exception as exc:
            if _err_mentions(exc, "not modified", "110026"):
                logger.debug("[%s] Margin mode already '%s' for %s", self.name, self._margin_mode, symbol)
            else:
                logger.warning("[%s] Could not set margin mode for %s: %s", self.name, symbol, exc)

        # ── Leverage (must be confirmed) ───────────────────────────────────────
        try:
            await self._client.set_leverage(self._leverage, mapped)
            self._leverage_set.add(symbol)
            logger.info("[%s] Leverage set to %dx for %s", self.name, self._leverage, symbol)
            return True
        except Exception as exc:
            # Bybit retCode 110043 "leverage not modified" → already at the target.
            if _err_mentions(exc, "leverage not modified", "110043"):
                self._leverage_set.add(symbol)
                logger.debug("[%s] Leverage already %dx for %s", self.name, self._leverage, symbol)
                return True
            logger.error(
                "[%s] Could not confirm leverage %dx for %s — SKIPPING order to avoid "
                "trading at unintended leverage: %s",
                self.name, self._leverage, symbol, exc,
            )
            return False

    async def _market_sell(self, symbol: str, amount: float) -> OrderResult:
        try:
            await self._ensure_leverage(symbol)
            logger.info("[%s] SELL %.6f %s", self.name, amount, symbol)
            # reduceOnly=True closes an existing long rather than opening a short
            params = {"reduceOnly": True} if self.market_type == "futures" else {}
            resp = await self._client.create_order(
                symbol=self._map_symbol(symbol),
                type="market", side="sell", amount=amount,
                params=params,
            )
            order_id = resp.get("id")
            logger.info("[%s] Sell placed: %s id=%s", self.name, symbol, order_id)
            fill_price, fill_cost, fill_amount = _extract_fill(resp, 0.0, amount)
            return OrderResult(
                exchange=self.name, symbol=symbol, side="sell",
                allocation=0.0, quantity=amount,
                order_id=str(order_id) if order_id else None, status="ok",
                price=fill_price or None, cost=fill_cost or None, filled=fill_amount or None,
            )
        except Exception as exc:
            logger.error("[%s] Sell failed for %s: %s", self.name, symbol, exc)
            return OrderResult(
                exchange=self.name, symbol=symbol, side="sell",
                allocation=0.0, quantity=amount, status="error", error=str(exc),
            )

    async def _market_buy(self, symbol: str, cost: float, allocation: float) -> OrderResult:
        try:
            # Opening/increasing a position: only proceed if leverage is confirmed.
            if not await self._ensure_leverage(symbol):
                return OrderResult(
                    exchange=self.name, symbol=symbol, side="buy",
                    allocation=allocation, quantity=0.0, status="error",
                    error=f"leverage not confirmed at {self._leverage}x — buy skipped",
                )
            price = (await self._client.fetch_ticker(self._map_symbol(symbol))).get("ask") or 0.0
            if not price:
                raise ValueError(f"No ask price for {symbol}")
            amount = cost / float(price)
            logger.info("[%s] BUY %.6f %s  (cost=%.4f, alloc=%.1f%%)",
                        self.name, amount, symbol, cost, allocation * 100)
            resp = await self._client.create_order(
                symbol=self._map_symbol(symbol),
                type="market", side="buy", amount=amount,
            )
            order_id = resp.get("id")
            logger.info("[%s] Buy placed: %s id=%s", self.name, symbol, order_id)
            fill_price, fill_cost, fill_amount = _extract_fill(resp, float(price), amount)
            return OrderResult(
                exchange=self.name, symbol=symbol, side="buy",
                allocation=allocation, quantity=amount,
                order_id=str(order_id) if order_id else None, status="ok",
                price=fill_price or None, cost=fill_cost or None, filled=fill_amount or None,
            )
        except Exception as exc:
            logger.error("[%s] Buy failed for %s: %s", self.name, symbol, exc)
            return OrderResult(
                exchange=self.name, symbol=symbol, side="buy",
                allocation=allocation, quantity=0.0, status="error", error=str(exc),
            )

    def _log_rebalance(
        self,
        held: Dict[str, float],
        quote_free: float,
        total_value: float,
        prices: Dict[str, float],
        quote: str,
        target: TargetPortfolio,
        sells: List[Tuple[str, float]],
        buys:  List[Tuple[str, float, float]],
    ) -> None:
        tag = f"[{self.name}]"
        sep = "─" * 48

        # Current portfolio (skip assets below 0.1% to avoid noise)
        current_lines = []
        for base, amount in sorted(held.items()):
            sym = f"{base}/{quote}"
            price = prices.get(sym, 0.0)
            value = amount * price
            pct = value / total_value if total_value > 0 else 0.0
            if pct >= 0.001:
                current_lines.append(f"  {base:<8} {pct:>6.1%}  ({amount:.6g} @ {price:.4g} = {value:.2f} {quote})")
        quote_pct = quote_free / total_value if total_value > 0 else 0.0
        if quote_pct >= 0.001:
            current_lines.append(f"  {quote:<8} {quote_pct:>6.1%}  (free: {quote_free:.2f} {quote})")

        # Target portfolio: signalling targets + protected positions held by silent systems
        target_entries: Dict[str, Tuple[float, str]] = {}  # base → (frac, label)
        for sym, frac in target.targets.items():
            target_entries[sym.split("/")[0]] = (frac, "")
        for base in target.protected_symbols:
            frac = target.protected_targets.get(base, 0.0)
            target_entries[base] = (frac, "  (held by other system)")
        for base, frac in target.protected_fraction.items():
            if base not in target_entries:
                target_entries[base] = (frac, "  (floor)")
        allocated = sum(frac for frac, _ in target_entries.values())
        cash_frac = max(0.0, 1.0 - allocated)
        if cash_frac >= 0.001:
            target_entries[quote] = (cash_frac, "  (cash)")
        target_lines = [
            f"  {base:<8} {frac:>6.1%}{label}"
            for base, (frac, label) in sorted(target_entries.items(), key=lambda x: -x[1][0])
        ]

        # Plan
        plan_lines = []
        for sym, amount in sells:
            value = amount * prices.get(sym, 0.0)
            plan_lines.append(f"  SELL {sym:<12} {amount:.6g}  (~{value:.2f} {quote})")
        for sym, cost, frac in buys:
            plan_lines.append(f"  BUY  {sym:<12} {cost:.2f} {quote}  ({frac:.1%} of portfolio)")

        mode = " [DRY RUN]" if self.dry_run else ""
        logger.info("%s %s  Total: %.2f %s%s", tag, sep, total_value, quote, mode)
        logger.info("%s Current allocation:", tag)
        for line in (current_lines or ["  (no holdings)"]): logger.info("%s%s", tag, line)
        logger.info("%s Target allocation:", tag)
        for line in (target_lines or ["  (empty — will sell everything)"]): logger.info("%s%s", tag, line)
        logger.info("%s Rebalance plan:", tag)
        for line in (plan_lines or ["  (nothing to trade — already at target or all trades below min_trade_value)"]): logger.info("%s%s", tag, line)
        logger.info("%s %s", tag, sep)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    async def simulate_rebalance(self, target: TargetPortfolio) -> List[OrderResult]:
        """Run rebalance in dry-run mode regardless of the client's configured dry_run flag."""
        original = self.dry_run
        self.dry_run = True
        try:
            return await self.rebalance(target)
        finally:
            self.dry_run = original

    async def close(self) -> None:
        await self._client.close()

    async def compute_equity(self) -> Dict[str, Any]:
        """Value the whole account in the quote currency: spot holdings + free cash + earn.

        Read-only — fetches balances and live prices but places no orders. Mirrors the
        valuation in _plan_rebalance (fetch_balance → _fetch_prices → amount × price).
        Returns a breakdown used by the equity sampler / dashboard / weekly report.
        """
        quote = self.quote_currency
        try:
            balance = await self.fetch_balance()
        except Exception as exc:
            logger.error("[%s] Failed to fetch balance for equity: %s", self.name, exc)
            return {
                "exchange": self.name, "quote": quote, "total_value": 0.0,
                "quote_free": 0.0, "earn_value": 0.0, "positions": [],
            }

        quote_bal = balance.get(quote)
        quote_free = float(quote_bal.get("total") or quote_bal.get("free") or 0.0) if isinstance(quote_bal, dict) else 0.0

        held: Dict[str, float] = {}
        for currency, bal in balance.items():
            if currency == quote or not isinstance(bal, dict):
                continue
            total = float(bal.get("total") or 0.0)
            if total > 0:
                held[currency] = total

        symbols_needed = {f"{base}/{quote}" for base in held}
        prices = await self._fetch_prices(symbols_needed) if symbols_needed else {}

        earn_value = 0.0
        try:
            earn_value = await self.fetch_earn_balance()
        except Exception as exc:
            logger.warning("[%s] Could not fetch earn balance for equity: %s", self.name, exc)

        positions: List[Dict[str, Any]] = []
        holdings_value = 0.0
        for base, amount in held.items():
            price = prices.get(f"{base}/{quote}", 0.0)
            value = amount * price
            holdings_value += value
            positions.append({"base": base, "amount": amount, "price": price, "value": value})

        total_value = quote_free + holdings_value + earn_value
        for p in positions:
            p["pct"] = (p["value"] / total_value) if total_value > 0 else 0.0
        positions.sort(key=lambda p: -p["value"])

        return {
            "exchange": self.name, "quote": quote, "total_value": total_value,
            "quote_free": quote_free, "earn_value": earn_value, "positions": positions,
        }

    async def fetch_earn_balance(self) -> float:
        """Return idle balance currently held in an earn/staking product (quote currency).

        Override in exchange subclasses that support earn (e.g. KrakenExchange).
        """
        return 0.0

    async def fetch_cashflows(self, since_ms: int) -> float:
        """Net external cashflow (deposits − withdrawals) in the quote currency since since_ms.

        Used to divide deposits/withdrawals out of the equity curve (TWR) so funding the
        account doesn't read as a gain. Uses ccxt's unified deposit/withdrawal history.

        # ponytail: quote-currency flows only — coin deposits would need a historical price
        # to value and are rare for these accounts. Add coin valuation if that changes.
        """
        quote = self.quote_currency
        net = 0.0
        fetchers = (
            ("fetchDeposits", getattr(self._client, "fetch_deposits", None), 1.0),
            ("fetchWithdrawals", getattr(self._client, "fetch_withdrawals", None), -1.0),
        )
        for cap, fn, sign in fetchers:
            if fn is None or not self._client.has.get(cap):
                continue
            try:
                txns = await fn(quote, since_ms)
            except Exception as exc:
                logger.warning("[%s] Could not fetch %s: %s", self.name, cap, exc)
                continue
            for t in txns:
                if t.get("currency") != quote:
                    continue
                if (t.get("status") or "ok") != "ok":  # skip pending/failed/canceled
                    continue
                net += sign * float(t.get("amount") or 0.0)
        return net

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Any]:
        """Return the account balance dict (ccxt format: {currency: {free, used, total}})."""
        ...
