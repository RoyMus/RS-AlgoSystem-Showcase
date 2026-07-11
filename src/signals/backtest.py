#!/usr/bin/env python3
"""
Backtest / validation tool for the RS Algo strategy.

Replays the SAME signal pipeline used live (engine.build_strategy) through a
daily equity simulation, so you can confirm the Python port matches your
TradingView "RS Algo System" script (net profit, max drawdown, monthly returns).

It reuses the engine's pure functions — nothing about the strategy is
reimplemented here, only the equity walk-forward + reporting.

Usage:
    python -m src.signals.backtest
    python -m src.signals.backtest --assets BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT
    python -m src.signals.backtest --start 2023-01-01 --no-plot
    python -m src.signals.backtest --assets BTCUSDT,ETHUSDT,SUIUSDT,BNBUSDT,HYPEUSDT

Outputs (written to ./backtest_out):
    equity.csv        daily equity, drawdown, peak
    daily_returns.csv daily strategy return (for diffing against a TV export)
    equity.png        equity + drawdown chart (only if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import engine

INITIAL_EQUITY = 1.0
ANN_DAYS       = 365            # crypto trades 24/7
OUT_DIR        = "backtest_out"
_MONTHS        = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ──────────────────────────────────────────────────────────────────────────────
# EQUITY SIMULATION  (mirrors Pine f_equity — [1]-lag timing)
# ──────────────────────────────────────────────────────────────────────────────

def run_equity_simulation(
    tickers:                  list[str],
    close_df:                 pd.DataFrame,
    best:                     pd.Series,
    second:                   pd.Series,
    equal_weight:             pd.Series,
    alloc_df:                 pd.DataFrame,
    bench_trend:              pd.Series,
    gold_trend:               pd.Series,
    start_date:               str,
    enable_total_filter:      bool = True,
    allow_gold_when_filtered: bool = True,
) -> pd.DataFrame:
    """Daily equity simulation. Timing: yesterday's (i-1) signal × today's return."""
    t7       = tickers[-1]
    returns  = close_df[tickers].pct_change()
    start_ts = pd.Timestamp(start_date)

    eq   = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    mdd  = 0.0
    records: list[dict] = []
    dates = close_df.index

    for i, date in enumerate(dates):
        if i > 0 and date >= start_ts:
            prev = dates[i - 1]

            b  = best.loc[prev]
            s  = second.loc[prev]
            ew = bool(equal_weight.loc[prev])
            bt = int(bench_trend.loc[prev])
            gt = int(gold_trend.loc[prev])

            filter_pass = (bt == 1) if enable_total_filter else True
            gold_ok     = allow_gold_when_filtered and (gt == 1)

            def _ret(ticker: str) -> float:
                if ticker in ("USD", "CASH") or ticker not in returns.columns:
                    return 0.0
                v = returns.loc[date, ticker]
                return float(v) if not np.isnan(v) else 0.0

            def _alloc(ticker: str) -> float:
                if ticker in ("USD", "CASH") or ticker not in alloc_df.columns:
                    return 1.0
                v = alloc_df.loc[prev, ticker]
                return float(v) if not np.isnan(v) else 0.8

            if filter_pass:
                ra = _ret(b)
                rb = _ret(s)
                wa = _alloc(b)
                wb = _alloc(s)

                invested  = (wa * 0.5 + wb * 0.5) if ew else wa
                ret_risky = (ra * wa * 0.5 + rb * wb * 0.5) if ew else ra * wa
                remainder = max(1.0 - invested, 0.0)

                paxg_in_port  = (b == t7) or (ew and s == t7)
                ret_remainder = _ret(t7) * remainder if (paxg_in_port or gt == 1) else 0.0

                eq *= 1.0 + ret_risky + ret_remainder

            elif gold_ok:
                eq *= 1.0 + _ret(t7)
            # else: 100% cash, no change

        peak = max(peak, eq)
        dd   = (peak - eq) / peak if peak > 0 else 0.0
        mdd  = max(mdd, dd)
        records.append({"date": date, "equity": eq, "mdd": round(mdd, 4), "peak": peak})

    return pd.DataFrame(records).set_index("date")


# ──────────────────────────────────────────────────────────────────────────────
# METRICS + REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(daily: pd.Series) -> dict:
    r = daily.dropna().to_numpy()
    n = len(r)
    if n == 0:
        return dict.fromkeys(("total", "cagr", "vol", "sharpe", "maxdd", "dd_days"), np.nan)
    eq    = np.cumprod(1.0 + r)
    total = eq[-1] - 1.0
    years = n / ANN_DAYS
    cagr  = eq[-1] ** (1.0 / years) - 1.0 if years > 0 and eq[-1] > 0 else np.nan
    sd    = r.std()
    vol   = sd * np.sqrt(ANN_DAYS)
    sharpe = r.mean() / sd * np.sqrt(ANN_DAYS) if sd > 0 else np.nan
    peak  = np.maximum.accumulate(eq)
    maxdd = ((peak - eq) / peak).max()
    below = eq < peak
    dd_days = run = 0
    for b in below:
        run = run + 1 if b else 0
        dd_days = max(dd_days, run)
    return {"total": total, "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "maxdd": maxdd, "dd_days": dd_days}


def _monthly_table(equity: pd.Series) -> None:
    """Print a TradingView-style year × month returns grid (+ YTD)."""
    m = equity.resample("ME").last().pct_change().dropna()
    by_yr: dict[int, dict[int, float]] = {}
    for ts, v in m.items():
        by_yr.setdefault(ts.year, {})[ts.month] = float(v)

    print("\n─── Monthly Returns (%) ─────────────────────────────────────────────────────────────────")
    print("  year " + "".join(f"{mo:>7s}" for mo in _MONTHS) + f"{'YTD':>8s}")
    for yr in sorted(by_yr):
        row = by_yr[yr]
        cells = "".join(f"{row[mo]*100:>7.1f}" if mo in row else f"{'':>7s}" for mo in range(1, 13))
        ytd = np.prod([1 + v for v in row.values()]) - 1.0
        print(f"  {yr} {cells}{ytd*100:>8.1f}")


def report(results: pd.DataFrame, s: engine.Strategy, config: dict,
           out_dir: str = OUT_DIR, make_plot: bool = True) -> None:
    start_ts = pd.Timestamp(config["start_date"])
    sim      = results.loc[results.index >= start_ts]
    equity   = sim["equity"]
    daily    = equity.pct_change().dropna()
    m        = _metrics(daily)

    # Buy-and-hold benchmark over the same window.
    bench_sym   = config["benchmark"]
    bench_close = s.closes[bench_sym].loc[s.closes.index >= equity.index[0]]
    bench_eq    = bench_close / bench_close.iloc[0]
    bench_total = float(bench_eq.iloc[-1] - 1.0)
    bench_dd    = float(((bench_eq.cummax() - bench_eq) / bench_eq.cummax()).max())

    name = engine.clean_name(bench_sym)
    print(f"\n─── Backtest Summary ({config['start_date']} → {equity.index[-1].date()}) ───")
    print(f"  Strategy total return : {m['total']*100:+.1f}%   (CAGR {m['cagr']*100:+.1f}%)")
    print(f"  HODL {name:<5s} return    : {bench_total*100:+.1f}%")
    print(f"  Final equity          : {equity.iloc[-1]:.4f}x")
    print(f"  Ann. volatility       : {m['vol']*100:.1f}%")
    print(f"  Sharpe (rf=0, √365)   : {m['sharpe']:.2f}")
    print(f"  Max drawdown          : {m['maxdd']*100:.1f}%   (HODL {name}: {bench_dd*100:.1f}%)")
    print(f"  Longest drawdown      : {m['dd_days']} days")

    _monthly_table(equity)

    # ── CSV exports (no extra deps; ideal for diffing against a TradingView export)
    os.makedirs(out_dir, exist_ok=True)
    sim.to_csv(os.path.join(out_dir, "equity.csv"))
    daily.rename("daily_return").to_csv(os.path.join(out_dir, "daily_returns.csv"))
    print(f"\n  → {out_dir}/equity.csv, {out_dir}/daily_returns.csv")

    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not installed — skipping equity.png; "
                  "`pip install -r requirements-backtest.txt` to enable)")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(equity.index, equity.values, color="#00bcd4", lw=1.5, label="RS Strategy")
        ax1.plot(bench_eq.index, bench_eq.values, color="#999", lw=1.0, label=f"HODL {name}")
        ax1.set_yscale("log"); ax1.set_ylabel("Equity (×)"); ax1.legend(); ax1.grid(alpha=0.3)
        ax1.set_title("RS Algo — backtest equity vs buy-and-hold")
        dd = (sim["equity"] - sim["peak"]) / sim["peak"] * 100
        ax2.fill_between(dd.index, dd.values, 0, color="#e74c3c", alpha=0.5)
        ax2.set_ylabel("Drawdown %"); ax2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "equity.png"), dpi=130)
        plt.close(fig)
        print(f"  → {out_dir}/equity.png")


def backtest(config: dict = engine.CONFIG):
    """Run the full pipeline + equity simulation. Returns (results_df, Strategy)."""
    s = engine.build_strategy(config)
    print("Simulating equity …")
    results = run_equity_simulation(
        s.all_assets, s.closes[s.all_assets], s.best, s.second, s.equal_weight,
        s.alloc_df, s.bench_trend, s.gold_trend,
        config["start_date"], config["enable_total_filter"], config["allow_gold_when_filtered"],
    )
    return results, s


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the RS Algo strategy.")
    ap.add_argument("--assets", help="comma-separated basket override, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--start", help="backtest start date (YYYY-MM-DD), overrides config")
    ap.add_argument("--out", default=OUT_DIR, help="output directory (default: backtest_out)")
    ap.add_argument("--no-plot", action="store_true", help="skip the equity.png chart")
    args = ap.parse_args()

    config = dict(engine.CONFIG)
    if args.assets:
        config["assets"] = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    if args.start:
        config["start_date"] = args.start

    results, s = backtest(config)
    report(results, s, config, out_dir=args.out, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
