"""Live execution smoke-test against a Bybit paper account (Demo Trading or Testnet).

Drives the real BybitExchange code path the same way the daily signal would, to
validate end-to-end execution before pointing the strategy at a real prop-firm
account. Set the DEMO / MARKET_TYPE knobs below:

  DEMO=True   → Demo Trading (api-demo.bybit.com) — supports spot AND futures.
  DEMO=False  → Testnet (api-testnet.bybit.com) — spot works; futures is
                KYC-blocked (retCode 10024) on the test account.

Flow:
  1. Show starting balance + open positions.
  2. Rebalance to a sample target (BTC/ETH) — places REAL paper orders.
  3. Show resulting positions.
  4. Flatten back to USDT (CASH target) — closes the positions.
  5. Show final state.

Run from the repo root:  python -m scripts.bybit_demo_test
Uses BYBIT_TEST_API_KEY / BYBIT_TEST_API_SECRET from .env.
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from src.execution.exchanges import BybitExchange
from src.models import TargetPortfolio

# ---- knobs ---------------------------------------------------------------
QUOTE = "USDT"
# DEMO=True  → mainnet Demo Trading (api-demo.bybit.com); supports derivatives.
# DEMO=False → public testnet (testnet.bybit.com); derivatives are KYC-blocked
#              (retCode 10024) on the test account, spot works.
DEMO = True
# "spot" or "futures" (linear USDT perps).
MARKET_TYPE = "futures"
# Sample target: deploy ~60% of equity split across two majors.
TARGET = {f"BTC/{QUOTE}": 0.30, f"ETH/{QUOTE}": 0.30}
FLATTEN_AT_END = True   # close everything back to USDT when done
# --------------------------------------------------------------------------


def _fmt_results(results) -> str:
    if not results:
        return "    (no orders)"
    lines = []
    for r in results:
        tag = "OK " if r.status == "ok" else r.status.upper()
        idpart = f" id={r.order_id}" if r.order_id else ""
        errpart = f"  ERROR: {r.error}" if r.error else ""
        lines.append(f"    [{tag}] {r.side:<4} {r.symbol:<14} qty={r.quantity:.6g}{idpart}{errpart}")
    return "\n".join(lines)


async def _show_state(ex: BybitExchange, label: str) -> None:
    print(f"\n=== {label} ===")
    eq = await ex.compute_equity()
    print(f"  total equity : {eq['total_value']:.2f} {eq['quote']}")
    print(f"  free quote   : {eq['quote_free']:.2f} {eq['quote']}")
    if MARKET_TYPE == "futures":
        pos = await ex._fetch_long_positions(
            TargetPortfolio(targets=TARGET, quote=QUOTE)
        )
        if pos:
            for base, contracts in pos.items():
                print(f"  position     : {base} {contracts:.6g} contracts (long)")
        else:
            print("  position     : (flat)")
    elif eq["positions"]:
        for p in eq["positions"]:
            print(f"  position     : {p['base']} {p['amount']:.6g}  "
                  f"(~{p['value']:.2f} {eq['quote']}, {p['pct']:.0%})")
    else:
        print("  position     : (flat)")


async def main() -> None:
    load_dotenv(os.path.abspath(".env"))
    key = os.getenv("BYBIT_TEST_API_KEY")
    secret = os.getenv("BYBIT_TEST_API_SECRET")
    if not key or not secret:
        raise SystemExit("BYBIT_TEST_API_KEY / BYBIT_TEST_API_SECRET not set in .env")

    ex = BybitExchange(
        api_key=key,
        api_secret=secret,
        testnet=not DEMO,      # testnet.bybit.com (when not in demo mode)
        demo=DEMO,             # api-demo.bybit.com (mainnet demo trading)
        dry_run=False,         # PLACE REAL (paper) ORDERS
        label="bybit_demo" if DEMO else "bybit_testnet",
        quote_currency=QUOTE,
        min_trade_value=5.0,
        order_delay_ms=1000,
        market_type=MARKET_TYPE,  # "spot" or "futures" (linear USDT perps)
        leverage=1,
        margin_mode="isolated",
    )

    try:
        await _show_state(ex, "START")

        print(f"\n>>> Rebalancing to target: "
              f"{ {k: f'{v:.0%}' for k, v in TARGET.items()} }")
        results = await ex.rebalance(TargetPortfolio(targets=TARGET, quote=QUOTE))
        print(_fmt_results(results))

        await asyncio.sleep(2)
        await _show_state(ex, "AFTER OPEN")

        if FLATTEN_AT_END:
            print("\n>>> Flattening (CASH target) — closing all positions")
            results = await ex.rebalance(TargetPortfolio(targets={}, quote=QUOTE))
            print(_fmt_results(results))
            await asyncio.sleep(2)
            await _show_state(ex, "FINAL")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
