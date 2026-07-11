#!/usr/bin/env python3
"""
Operate & verify the running app via its API (local or production).

    python scripts/control.py status      # health + equity + current allocation
    python scripts/control.py equity       # equity curve summary + P&L
    python scripts/control.py positions    # current portfolio allocation
    python scripts/control.py profit       # per-position unrealized P&L vs cost basis
    python scripts/control.py state        # in-memory signal state per system
    python scripts/control.py run          # run the RS generator NOW + show scores
    python scripts/control.py preview "BTC/USD 50% | ETH/USD 50%"   # dry-run rebalance
    python scripts/control.py report       # send the weekly Telegram report now
    python scripts/control.py reset-equity # clear equity history to re-baseline the curve

Options (or env):
    --url    base URL  (env CTRL_URL, default https://signalautomation.fly.dev)
    --token  webhook/admin token (env WEBHOOK_TOKEN; also read from .env)

Note: `run` triggers a REAL rebalance if the allocation changed. `preview` never
places orders. status/equity/positions are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEFAULT_URL = os.getenv("CTRL_URL", "https://signalautomation.fly.dev")


def call(method: str, url: str, token: str = "", body: dict | None = None, auth: bool = True) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth and token:
        headers["X-Token"] = token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=320) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        sys.exit(f"ERROR: {method} {url} -> HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: cannot reach {url}: {e.reason}")


def show_equity(base: str) -> None:
    eq = call("GET", base + "/public/equity", auth=False)
    pts = eq.get("points") or []
    q = eq.get("quote") or ""
    if not pts:
        print("No equity data yet.")
        return
    first, last = pts[0]["total_value"], pts[-1]["total_value"]
    chg = last - first if (first is not None and last is not None) else None
    pct = (chg / first * 100) if (chg is not None and first) else None
    print(f"Equity     : {last:.2f} {q}   ({len(pts)} samples)")
    print(f"First      : {first:.2f} {q}   @ {pts[0]['ts']}")
    print(f"Latest     : {last:.2f} {q}   @ {pts[-1]['ts']}")
    if chg is not None:
        print(f"Change     : {chg:+.2f} {q}  ({pct:+.2f}%)")


def show_positions(base: str) -> None:
    pos = call("GET", base + "/public/positions", auth=False)
    q = pos.get("quote") or ""
    tv = pos.get("total_value")
    print(f"As of      : {pos.get('timestamp')}")
    print(f"Total value: {tv if tv is None else f'{tv:.2f}'} {q}")
    rows = list(pos.get("positions") or [])
    if pos.get("quote_free", 0) >= 0.01:
        rows.append({"base": "Cash", "value": pos["quote_free"]})
    if pos.get("earn_value", 0) >= 0.01:
        rows.append({"base": "Earn", "value": pos["earn_value"]})
    if not rows:
        print("  (no positions)")
        return
    print(f"  {'ASSET':<8}{'VALUE':>14}{'ALLOC':>9}")
    for r in sorted(rows, key=lambda x: -(x.get("value") or 0)):
        v = r.get("value") or 0
        pct = (v / tv * 100) if tv else 0
        print(f"  {r['base']:<8}{v:>14.2f}{pct:>8.1f}%")


def show_profit(base: str, token: str) -> None:
    data = call("GET", base + "/public/profit", token=token)
    rows = data.get("positions") or []
    q = data.get("quote") or ""
    if not rows:
        print("No open positions.")
        return
    print(f"  {'ASSET':<8}{'PROFIT':>9}{'ENTRY':>12}{'PRICE':>12}{'VALUE':>14}{'ALLOC':>8}")
    for r in rows:
        pp = r.get("profit_pct")
        pp_s = f"{pp:+.1f}%" if pp is not None else "n/a"
        entry = r.get("avg_entry")
        entry_s = f"{entry:.4g}" if entry else "?"
        print(
            f"  {r.get('base',''):<8}{pp_s:>9}{entry_s:>12}"
            f"{r.get('price', 0):>12.4g}{r.get('value', 0):>14.2f}"
            f"{(r.get('pct') or 0)*100:>7.1f}%"
        )


def show_run(base: str, token: str) -> None:
    print("Running RS generator now (may take up to ~5 min) ...")
    res = call("POST", base + "/signal/run", token=token)
    for inst in res.get("instances", []):
        print(f"\n=== {inst['system']}  ({inst['date']}) ===")
        print(f"  Allocation: {inst['allocation']}")
        print(f"  Webhook   : {inst['webhook']}")
        scores = inst.get("scores") or {}
        if scores:
            print(f"  RS scores (selection ranking):")
            print(f"    {'TICKER':<10}{'SCORE':>8}{'ALPHA':>10}{'TREND':>8}")
            def _sc(v):
                return v.get("score", 0) if isinstance(v, dict) else 0
            for tkr, sv in sorted(scores.items(), key=lambda kv: -_sc(kv[1])):
                if isinstance(sv, dict):
                    a = sv.get("alpha")
                    print(f"    {tkr:<10}{_sc(sv):>8}{(a if a is not None else 0):>10.3f}{sv.get('trend', 0):>8}")
    print("\n(If allocation changed vs last run, a real rebalance was enqueued.)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "equity", "positions", "profit", "state", "run", "preview", "report", "reset-equity"])
    p.add_argument("allocations", nargs="?", help="for `preview`: e.g. 'BTC/USD 50% | ETH/USD 50%'")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--token", default=os.getenv("WEBHOOK_TOKEN", ""))
    args = p.parse_args()
    base = args.url.rstrip("/")

    needs_token = args.command in ("state", "run", "preview", "report", "reset-equity", "profit")
    if needs_token and not args.token:
        sys.exit("ERROR: no token. Pass --token or set WEBHOOK_TOKEN (env or .env).")

    if args.command == "status":
        h = call("GET", base + "/health", auth=False)
        print(f"Health     : {h.get('status')}  exchanges={h.get('exchanges')}  signals={h.get('signals')}")
        show_equity(base)
        show_positions(base)
    elif args.command == "equity":
        show_equity(base)
    elif args.command == "positions":
        show_positions(base)
    elif args.command == "profit":
        show_profit(base, args.token)
    elif args.command == "state":
        print(json.dumps(call("GET", base + "/state", token=args.token), indent=2))
    elif args.command == "run":
        show_run(base, args.token)
    elif args.command == "preview":
        body = {"system": "system_1", "allocations": args.allocations} if args.allocations else None
        print(json.dumps(call("POST", base + "/test/rebalance", token=args.token, body=body), indent=2))
    elif args.command == "report":
        print(json.dumps(call("POST", base + "/report/now", token=args.token), indent=2))
    elif args.command == "reset-equity":
        print(json.dumps(call("POST", base + "/admin/equity/reset", token=args.token), indent=2))


if __name__ == "__main__":
    main()
