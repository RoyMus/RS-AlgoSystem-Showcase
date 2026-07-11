#!/usr/bin/env python3
"""
RS Algo System (Crypto) — simplified showcase edition.

A relative-strength rotation over a fixed crypto basket, with a spot-gold trend
filter for bear markets. This is the signal generator only: it fetches live
prices, computes the strategy signals, and prints the current target allocation
plus the webhook payload to send to the execution server.

Simplified vs. the production strategy:
  - Signals are RSI-based (RSI(14) on a 7-day EMA, +1 above 50 / -1 below),
    replacing the original scoring-CCI indicator.
  - Fixed basket only — no dynamic asset-selection engine.
  - Fixed position sizing (no volatility-parity "dynamic hedging").

Universe:
    Risky : BTC, ETH, SOL, SUI, XRP, BNB   (USDT pairs)
    Hedge : PAXG  (gold proxy, == t7)      — held in bear markets
    Bench : BTC   (alpha / trend filter)
    Gold trend filter : GLD (SPDR physical-gold ETF, weekday close)

Dependencies:
    pip install requests pandas numpy yfinance

Usage:
    python -m src.signals.engine
"""

import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and choke on box-drawing glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ALPHA_LEN     = 35
EPS           = 1e-10
RSI_LEN       = 14      # RSI lookback
EMA_LEN       = 7       # EMA smoothing applied before RSI
RSI_THRESH    = 50.0    # RSI > 50 → +1, else -1
HEDGE_WEIGHT  = 0.8     # fixed fraction into the top RS asset (no vol-parity sizing)

# Binance interval string for each supported bar_interval.
_BINANCE_INTERVAL = {"1d": "1d", "1wk": "1w", "1mo": "1M"}

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CONFIG: dict = {
    # Risky basket (Binance USDT pairs).
    "assets":                    ["BTCUSDT", "ETHUSDT", "SOLUSDT",
                                  "SUIUSDT", "XRPUSDT", "BNBUSDT"],
    "gold_sym":                  "PAXGUSDT",      # held hedge asset (t7)
    # Gold *trend filter* source. GLD (SPDR physical-gold ETF, weekday close)
    # is a clean spot-gold proxy: its Mon–Fri series held flat over weekends
    # flips the bear-market filter far less than PAXG's 24/7 price. Set to None
    # to fall back to PAXG's own trend, or any yfinance symbol (e.g. "GC=F").
    "gold_filter_sym":           "GLD",
    "benchmark":                 "BTCUSDT",       # alpha / trend filter

    "start_date":                "2023-01-01",    # window start (for context only)
    "data_start":                "2021-01-01",    # earliest data to request
    "bar_interval":              "1d",            # "1d" | "1wk" | "1mo"

    "alpha_len":                 ALPHA_LEN,
    "enable_total_filter":       True,            # bear-market override on/off
    "allow_gold_when_filtered":  True,            # hold gold (not cash) in a filtered bear market
    "system_id":                 1,
    "webhook_secret":            "1234",

    # Public Binance market-data endpoint (no API key, not geo-blocked).
    "binance_base":              "https://data-api.binance.vision",

    # Symbols NOT listed on Binance → fetch daily closes from another venue.
    # Map the basket symbol to its source spec. Only consulted for symbols you
    # actually put in a basket.
    "data_sources": {
        "HYPEUSDT": {"source": "hyperliquid", "coin": "HYPE"},
        "XMRUSDT":  {"source": "kraken", "pair": "XMRUSD"},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def clean_name(sym: str) -> str:
    """Short display name from a Binance symbol, e.g. BTCUSDT → BTC, PAXGUSDT → PAXG."""
    if sym in ("USD", "USDT", "CASH"):
        return "CASH"
    s = sym
    for suf in ("USDT", "USDC", "USD"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def webhook_name(sym: str) -> str:
    if sym in ("USD", "USDT", "CASH"):
        return "USDT"
    return clean_name(sym) + "/USDT"


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (RSI-based relative strength)
# ──────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = RSI_LEN) -> pd.Series:
    """Wilder's RSI in [0, 100]. NaN during the warmup window."""
    delta = series.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Special cases, applied only on warmed-up bars (NaN warmup stays NaN):
    #   pure uptrend (no losses) → 100; perfectly flat → 50.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def rsi_signal(series: pd.Series) -> pd.Series:
    """+1 when RSI(EMA(series, 7), 14) > 50, -1 when below; 0 during warmup."""
    smooth = series.ewm(span=EMA_LEN, adjust=False).mean()
    r = _rsi(smooth, RSI_LEN)
    sig = pd.Series(0, index=series.index, dtype=int)
    sig[r > RSI_THRESH]  = 1
    sig[r <= RSI_THRESH] = -1
    # Keep warmup bars (RSI NaN) neutral rather than short.
    sig[r.isna()] = 0
    return sig


def f_ratios(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pairwise relative-strength signals: is a strengthening vs b, and vice-versa."""
    ratio_ab = a / b.replace(0.0, np.nan)
    ratio_ba = b / a.replace(0.0, np.nan)
    return rsi_signal(ratio_ab), rsi_signal(ratio_ba)


def f_trend(close: pd.Series) -> pd.Series:
    return rsi_signal(close)


def f_intra_trend(close: pd.Series) -> pd.Series:
    return rsi_signal(close)


def jensen_alpha(
    asset_close: pd.Series,
    bench_close: pd.Series,
    period: int = ALPHA_LEN,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Rolling Jensen's alpha of the asset vs the benchmark."""
    r_a = asset_close.pct_change()
    r_b = bench_close.pct_change()

    mA  = r_a.rolling(period).mean()
    mB  = r_b.rolling(period).mean()
    mAB = (r_a * r_b).rolling(period).mean()
    mBB = (r_b * r_b).rolling(period).mean()

    cov   = mAB - mA * mB
    var_b = mBB - mB * mB
    beta1 = (cov / var_b).where(var_b > EPS)

    corr  = r_a.rolling(period).corr(r_b)
    sd_a  = r_a.rolling(period).std()
    sd_b  = r_b.rolling(period).std()
    beta2 = (corr * sd_a / sd_b).where(sd_b > EPS)

    beta = beta1.combine_first(beta2).fillna(0.0)
    return mA - beta * mB


def f_allocation(index: pd.DatetimeIndex) -> pd.Series:
    """Fixed position sizing — HEDGE_WEIGHT into the top RS asset, remainder to
    cash/gold. (No volatility-parity dynamic hedging in the simplified version.)"""
    return pd.Series(HEDGE_WEIGHT, index=index)


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING  (Binance public klines)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_klines(base: str, symbol: str, interval: str, start_ms: int) -> list:
    """Paginated klines download (1000 bars/request) from start_ms to now."""
    url = f"{base}/api/v3/klines"
    out: list = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 400:
            # Symbol not listed / invalid pair — treat as no data.
            return out
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out += data
        if len(data) < 1000:
            break
        cur = data[-1][0] + 1
        time.sleep(0.15)
    return out


def fetch_closes(
    symbols: list[str],
    start: str,
    interval: str = "1d",
    base: str = "https://data-api.binance.vision",
) -> pd.DataFrame:
    """
    Download daily close prices for each Binance symbol and return a DataFrame
    keyed by symbol. Forward-fills internal gaps; leading NaNs (pre-listing) remain.
    """
    bin_interval = _BINANCE_INTERVAL.get(interval, "1d")
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)

    series: dict[str, pd.Series] = {}
    for sym in symbols:
        kl = _fetch_klines(base, sym, bin_interval, start_ms)
        if not kl:
            print(f"  ! no data for {sym}")
            continue
        ts    = pd.to_datetime([k[0] for k in kl], unit="ms")
        close = pd.Series([float(k[4]) for k in kl], index=ts, name=sym)
        # Collapse to one row per calendar day (drop intra-period dupes from pagination).
        close = close[~close.index.duplicated(keep="last")]
        series[sym] = close
        print(f"  {sym:10s} {len(close):5d} bars  {close.index[0].date()} → {close.index[-1].date()}")

    closes = pd.DataFrame(series).sort_index()
    # ffill weekend/holiday-style gaps but keep pre-listing leading NaNs.
    return closes.ffill().dropna(how="all")


def fetch_kraken_closes(pairs: dict[str, str], interval: str = "1d") -> pd.DataFrame:
    """Daily closes from Kraken's public OHLC API for symbols not on Binance.

    `pairs` maps the basket symbol (e.g. "XMRUSDT") → Kraken pair (e.g. "XMRUSD").
    Kraken returns at most ~720 candles per request.
    """
    interval_min = {"1d": 1440, "1wk": 10080}.get(interval, 1440)
    series: dict[str, pd.Series] = {}
    for bsym, kpair in pairs.items():
        try:
            r = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": kpair, "interval": interval_min}, timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            print(f"  ! kraken fetch failed for {bsym} ({kpair}): {e}")
            continue
        if payload.get("error"):
            print(f"  ! kraken error for {bsym} ({kpair}): {payload['error']}")
            continue
        result = payload.get("result", {})
        rows_key = next((k for k in result if k != "last"), None)
        if not rows_key or not result[rows_key]:
            print(f"  ! no Kraken data for {bsym} ({kpair})")
            continue
        rows = result[rows_key]
        ts    = pd.to_datetime([row[0] for row in rows], unit="s").normalize()
        close = pd.Series([float(row[4]) for row in rows], index=ts, name=bsym)
        close = close[~close.index.duplicated(keep="last")]
        series[bsym] = close
        print(f"  {bsym:10s} {len(close):5d} bars  {close.index[0].date()} → {close.index[-1].date()}  (Kraken {kpair})")
    return pd.DataFrame(series).sort_index()


def fetch_hyperliquid_closes(coins: dict[str, str], start: str, interval: str = "1d") -> pd.DataFrame:
    """Daily closes from the Hyperliquid public API for symbols not on Binance.

    `coins` maps the basket symbol (e.g. "HYPEUSDT") → Hyperliquid coin (e.g. "HYPE").
    """
    hl_interval = {"1d": "1d", "1wk": "1w", "1mo": "1M"}.get(interval, "1d")
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    series: dict[str, pd.Series] = {}
    for bsym, coin in coins.items():
        try:
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "candleSnapshot",
                      "req": {"coin": coin, "interval": hl_interval,
                              "startTime": start_ms, "endTime": end_ms}},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ! hyperliquid fetch failed for {bsym} ({coin}): {e}")
            continue
        if not data:
            print(f"  ! no Hyperliquid data for {bsym} ({coin})")
            continue
        ts    = pd.to_datetime([c["t"] for c in data], unit="ms").normalize()
        close = pd.Series([float(c["c"]) for c in data], index=ts, name=bsym)
        close = close[~close.index.duplicated(keep="last")]
        series[bsym] = close
        print(f"  {bsym:10s} {len(close):5d} bars  {close.index[0].date()} → {close.index[-1].date()}  (Hyperliquid {coin})")
    return pd.DataFrame(series).sort_index()


def fetch_external_closes(specs: dict[str, dict], start: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch daily closes for non-Binance symbols, dispatching per `source`."""
    by_source: dict[str, dict] = {}
    for bsym, spec in specs.items():
        by_source.setdefault(spec.get("source"), {})[bsym] = spec

    frames: list[pd.DataFrame] = []
    if "hyperliquid" in by_source:
        coins = {b: s["coin"] for b, s in by_source.pop("hyperliquid").items()}
        frames.append(fetch_hyperliquid_closes(coins, start, interval))
    if "kraken" in by_source:
        pairs = {b: s["pair"] for b, s in by_source.pop("kraken").items()}
        frames.append(fetch_kraken_closes(pairs, interval))
    if by_source:
        print(f"  ! unknown data source(s): {sorted(by_source)}")

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.sort_index()


def fetch_gold_filter(
    sym: str,
    index: pd.DatetimeIndex,
    start: str,
    interval: str = "1d",
) -> pd.Series:
    """
    Spot-gold close series for the trend filter, reindexed onto the crypto
    calendar and forward-filled. Gold's Mon–Fri series held flat over weekends
    flips the bear-market filter far less than PAXG's 24/7 price.
    """
    import yfinance as yf  # local import: only needed when a gold filter is set
    yf_interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}.get(interval, "1d")
    df = yf.download(sym, start=start, auto_adjust=True, progress=False, interval=yf_interval)
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    out = c.reindex(index).ffill()
    print(f"  gold filter {sym:8s} {int(c.notna().sum()):5d} obs  {c.index[0].date()} → {c.index[-1].date()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ASSET SELECTION  (best / second RS asset per bar)
# ──────────────────────────────────────────────────────────────────────────────

def select_assets(
    pairwise:     dict[tuple[str, str], pd.Series],
    trends:       dict[str, pd.Series],
    alphas:       dict[str, pd.Series],
    get_universe: Callable,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Per-bar asset selection by relative-strength score.

    RS scores are computed per date against only the active (listed) universe.
    Disqualifies assets with: trend < 0, NaN alpha, or alpha ≤ median alpha.
    Returns (best, second, equal_weight) Series indexed by date.
    """
    idx = next(iter(trends.values())).index
    best_list:   list[str]  = []
    second_list: list[str]  = []
    ew_list:     list[bool] = []

    for date in idx:
        active = get_universe(date)

        raw_scores: dict[str, int] = {t: 0 for t in active}
        for t_a in active:
            for t_b in active:
                if t_a == t_b:
                    continue
                sig = pairwise.get((t_a, t_b))
                if sig is not None:
                    v = sig.loc[date]
                    raw_scores[t_a] += int(v) if not np.isnan(v) else 0

        trend_row = {t: int(trends[t].loc[date]) if t in trends else 0  for t in active}
        alpha_row = {t: float(alphas[t].loc[date]) if t in alphas else np.nan for t in active}

        alpha_vals = [v for v in alpha_row.values() if not np.isnan(v)]
        med        = float(np.median(alpha_vals)) if alpha_vals else 0.0

        eff: dict[str, float] = {}
        for t in active:
            tr = trend_row[t]
            al = alpha_row[t]
            if tr < 0 or np.isnan(al) or al <= med:
                eff[t] = -999_999.0
            else:
                eff[t] = raw_scores[t]

        sorted_t = sorted(active, key=lambda t: eff[t], reverse=True)
        best_t   = sorted_t[0]
        no_valid = eff[best_t] == -999_999.0

        if no_valid:
            best_list.append("USD")
            second_list.append("USD")
            ew_list.append(False)
            continue

        best_list.append(best_t)

        sec_t = next((t for t in sorted_t[1:] if eff[t] != -999_999.0), None)

        if sec_t is None:
            second_list.append("USD")
            ew_list.append(False)
        else:
            is_equal = eff[best_t] == eff[sec_t]
            if is_equal and trend_row.get(sec_t, -1) != 1:
                is_equal = False
            second_list.append(sec_t)
            ew_list.append(is_equal)

    return (
        pd.Series(best_list,   index=idx, name="best"),
        pd.Series(second_list, index=idx, name="second"),
        pd.Series(ew_list,     index=idx, name="equal_weight"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# ALLOCATION DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def build_alloc_display(
    best:          str,
    second:        str,
    ew:            bool,
    wa:            float,
    wb:            float,
    t7:            str,
    gold_positive: bool,
) -> dict[str, float]:
    """Compute display percentage weights for the current allocation."""
    invested  = (wa * 0.5 + wb * 0.5) if ew else wa
    remainder = max(1.0 - invested, 0.0)

    paxg_is_best   = best == t7
    paxg_is_second = ew and second == t7
    paxg_in_port   = paxg_is_best or paxg_is_second

    if paxg_is_best:
        paxg_rs_pct = wa * 50.0 if ew else wa * 100.0
    elif paxg_is_second:
        paxg_rs_pct = wb * 50.0
    else:
        paxg_rs_pct = 0.0

    paxg_remainder_pct = remainder * 100.0 if (paxg_in_port or gold_positive) else 0.0
    cash_remainder_pct = remainder * 100.0 if (not paxg_in_port and not gold_positive) else 0.0

    best_pct   = 0.0 if paxg_is_best   else (wa * 50.0 if ew else wa * 100.0)
    second_pct = 0.0 if paxg_is_second else (wb * 50.0 if ew else 0.0)

    return {
        "best_pct":       best_pct,
        "second_pct":     second_pct,
        "paxg_total_pct": paxg_rs_pct + paxg_remainder_pct,
        "cash_pct":       cash_remainder_pct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY PIPELINE  (shared by the live signal and the backtest)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """Everything the strategy produces, up to (and including) per-bar selection.

    Consumed by both compute_signal() (live, looks at the last bar) and the
    backtest equity simulation (replays the whole series), so the two can never
    diverge — they run on the exact same `best` / `second` / `equal_weight`.
    """
    config:       dict
    closes:       pd.DataFrame
    all_assets:   list[str]
    t7:           str
    bench_trend:  pd.Series
    gold_trend:   pd.Series
    intra_trends: dict[str, pd.Series]
    alphas:       dict[str, pd.Series]
    pairwise:     dict[tuple[str, str], pd.Series]
    alloc_df:     pd.DataFrame
    best:         pd.Series
    second:       pd.Series
    equal_weight: pd.Series
    get_universe: Callable[[pd.Timestamp], list[str]]


def build_strategy(config: dict = CONFIG) -> Strategy:
    """Fetch data and compute signals → selection (incl. the bear-market override).

    This is the single source of truth for the strategy logic. Returns a Strategy
    bundle; nothing about the *current* bar or webhook is decided here.
    """
    assets    = list(config["assets"])
    gold_sym  = config["gold_sym"]
    bench_sym = config["benchmark"]
    t7        = gold_sym
    interval  = config.get("bar_interval", "1d")
    base      = config.get("binance_base", "https://data-api.binance.vision")

    # ── Fetch data ─────────────────────────────────────────────────────────────
    all_syms = list(dict.fromkeys(assets + [gold_sym, bench_sym]))
    data_sources = config.get("data_sources", {}) or {}
    external     = {s: data_sources[s] for s in all_syms if s in data_sources}
    binance_syms = [s for s in all_syms if s not in data_sources]

    print("Fetching data from Binance …")
    closes = fetch_closes(binance_syms, config["data_start"], interval, base)

    if external:
        print("Fetching non-Binance symbols …")
        ext = fetch_external_closes(external, config["data_start"], interval)
        if not ext.empty:
            closes = closes.join(ext, how="outer").sort_index().ffill()
    closes = closes.loc[closes.index >= pd.Timestamp(config["data_start"])].dropna(how="all")
    available = set(closes.columns)

    if bench_sym not in available:
        raise RuntimeError(f"Benchmark {bench_sym} unavailable — cannot proceed.")

    bench_close = closes[bench_sym]

    # Gold *trend filter*: spot-gold proxy (e.g. GLD) if configured, else PAXG's
    # own price. PAXG (gold_sym) stays the held hedge asset regardless.
    gold_filter_sym = config.get("gold_filter_sym")
    if gold_filter_sym and gold_filter_sym != gold_sym:
        try:
            gold_close = fetch_gold_filter(gold_filter_sym, closes.index, config["data_start"], interval)
        except Exception as e:
            print(f"  ! gold filter {gold_filter_sym} failed ({e}); falling back to PAXG trend")
            gold_close = closes[gold_sym] if gold_sym in available else pd.Series(np.nan, index=closes.index)
    else:
        gold_close = closes[gold_sym] if gold_sym in available else pd.Series(np.nan, index=closes.index)

    # All selectable symbols, gold always last so that t7 == all_assets[-1].
    risky_assets = [t for t in assets if t in available]
    all_assets   = [t for t in risky_assets if t != gold_sym] + [gold_sym]

    def get_universe(date: pd.Timestamp) -> list[str]:
        """Active (already-listed) assets at `date`, gold hedge last."""
        risky = [t for t in assets if t in available and not np.isnan(closes.loc[date, t])]
        out   = risky[:]
        if gold_sym in available and not np.isnan(closes.loc[date, gold_sym]):
            out.append(gold_sym)
        return out or [gold_sym]

    # ── Signals ────────────────────────────────────────────────────────────────
    print("Computing signals …")
    bench_trend = f_trend(bench_close)
    gold_trend  = f_trend(gold_close)

    intra_trends: dict[str, pd.Series] = {t: f_intra_trend(closes[t]) for t in all_assets}
    alphas: dict[str, pd.Series] = {
        t: jensen_alpha(
            closes[t], bench_close,
            risk_free_rate=0.05 / 365 if t == t7 else 0.0,
        )
        for t in all_assets
    }

    print("Computing pairwise RS signals …")
    pairwise: dict[tuple[str, str], pd.Series] = {}
    for i, t_a in enumerate(all_assets):
        for t_b in all_assets[i + 1:]:
            sig_ab, sig_ba = f_ratios(closes[t_a], closes[t_b])
            pairwise[(t_a, t_b)] = sig_ab
            pairwise[(t_b, t_a)] = sig_ba

    # ── Fixed allocations (no vol-parity dynamic hedging) ───────────────────────
    alloc_df = pd.DataFrame(
        {t: f_allocation(closes.index) for t in all_assets},
        index=closes.index,
    )

    # ── Asset selection ────────────────────────────────────────────────────────
    print("Selecting assets …")
    best, second, equal_weight = select_assets(pairwise, intra_trends, alphas, get_universe)

    # Benchmark-trend override (bear-market filter)
    bear_mask = bool(config["enable_total_filter"]) & (bench_trend == -1)
    for date in closes.index[bear_mask]:
        gold_ok = config["allow_gold_when_filtered"] and gold_trend.loc[date] == 1
        best.loc[date]         = t7 if gold_ok else "USD"
        second.loc[date]       = "USD"
        equal_weight.loc[date] = False

    return Strategy(
        config=config, closes=closes, all_assets=all_assets, t7=t7,
        bench_trend=bench_trend, gold_trend=gold_trend,
        intra_trends=intra_trends, alphas=alphas, pairwise=pairwise,
        alloc_df=alloc_df, best=best, second=second, equal_weight=equal_weight,
        get_universe=get_universe,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LIVE SIGNAL  (current allocation + webhook payload)
# ──────────────────────────────────────────────────────────────────────────────

def compute_signal(config: dict = CONFIG) -> dict:
    """Build the strategy and return the *current* allocation + webhook payload:
        {
          "date": <last bar date>,
          "allocation": "BTC 80.0% | PAXG 20.0%",   # human display
          "webhook": "BTC/USDT 80.0% | PAXG/USDT 20.0%",
          "payload": {"system": "system_1", "allocations": "...", "token": "..."},
          "scores": {ticker: {"score", "alpha", "trend"}},
        }
    """
    s            = build_strategy(config)
    closes       = s.closes
    t7           = s.t7
    gold_trend   = s.gold_trend
    intra_trends = s.intra_trends
    alphas       = s.alphas
    pairwise     = s.pairwise
    alloc_df     = s.alloc_df
    best, second, equal_weight = s.best, s.second, s.equal_weight
    get_universe = s.get_universe

    # ── Current allocation ─────────────────────────────────────────────────────
    last_date  = closes.index[-1]
    cur_best   = str(best.iloc[-1])
    cur_second = str(second.iloc[-1])
    cur_ew     = bool(equal_weight.iloc[-1])
    cur_gold   = int(gold_trend.iloc[-1]) == 1

    wa = float(alloc_df.loc[last_date, cur_best])   if cur_best   in alloc_df.columns else 1.0
    wb = float(alloc_df.loc[last_date, cur_second]) if cur_second in alloc_df.columns else 1.0

    alloc = build_alloc_display(cur_best, cur_second, cur_ew, wa, wb, t7, cur_gold)

    # ── RS scores (latest bar) ─────────────────────────────────────────────────
    cur_universe = get_universe(last_date)
    last_rs: dict[str, int] = {t: 0 for t in cur_universe}
    for t_a in cur_universe:
        for t_b in cur_universe:
            if t_a != t_b:
                sig = pairwise.get((t_a, t_b))
                if sig is not None:
                    v = sig.iloc[-1]
                    last_rs[t_a] += int(v) if not np.isnan(v) else 0

    print("\n─── RS Scores (latest bar) ─────────────────────────────")
    print(f"  {'':8s}  {'score':>6s}  {'alpha':>10s}  {'trend':>5s}")
    scores: dict[str, dict] = {}
    for t in cur_universe:
        score = last_rs[t]
        alpha = alphas[t].iloc[-1] if t in alphas else float("nan")
        trend = int(intra_trends[t].iloc[-1]) if t in intra_trends else 0
        flag  = "*" if t == cur_best else ("+" if cur_ew and t == cur_second else " ")
        alpha_str = f"{alpha:+.6f}" if not np.isnan(alpha) else "     nan"
        print(f"  {flag} {clean_name(t):6s}  {score:+6d}  {alpha_str}  {trend:+5d}")
        scores[clean_name(t)] = {"score": score, "alpha": float(alpha), "trend": trend}

    # "best"/"second" can be the cash sentinel ("USD") when the bear filter is on;
    # cash must NOT appear as a tradable leg — the executor holds any uninvested
    # remainder as cash, and an all-cash signal is emitted as the "CASH" keyword.
    _CASH = ("USD", "USDT", "CASH")
    best_tradable   = cur_best   not in _CASH and cur_best   != t7 and alloc["best_pct"]   > 0
    second_tradable = cur_ew and cur_second not in _CASH and cur_second != t7 and alloc["second_pct"] > 0

    # ── Human-readable allocation ──────────────────────────────────────────────
    parts: list[str] = []
    if best_tradable:
        parts.append(f"{clean_name(cur_best)} {alloc['best_pct']:.1f}%")
    if second_tradable:
        parts.append(f"{clean_name(cur_second)} {alloc['second_pct']:.1f}%")
    if alloc["paxg_total_pct"] > 0:
        parts.append(f"{clean_name(t7)} {alloc['paxg_total_pct']:.1f}%")
    if alloc["cash_pct"] > 0:
        parts.append(f"CASH {alloc['cash_pct']:.1f}%")

    alloc_display = " | ".join(parts) if parts else "CASH"
    print(f"\n─── Current Allocation ({last_date.date()}) ───────────────────")
    print(f"  {alloc_display}")

    # ── Webhook payload ────────────────────────────────────────────────────────
    wh_parts: list[str] = []
    if best_tradable:
        wh_parts.append(f"{webhook_name(cur_best)} {round(alloc['best_pct'], 1)}%")
    if second_tradable:
        wh_parts.append(f"{webhook_name(cur_second)} {round(alloc['second_pct'], 1)}%")
    if alloc["paxg_total_pct"] > 0:
        wh_parts.append(f"{webhook_name(t7)} {round(alloc['paxg_total_pct'], 1)}%")
    wh_str = " | ".join(wh_parts) if wh_parts else "CASH"
    sys_id = config.get("system_id", 1)
    secret = config.get("webhook_secret", "")

    payload = {"system": f"system_{sys_id}", "allocations": wh_str, "token": secret}

    print(f"\n─── Webhook Payload ────────────────────────────────────")
    print(f'  {{"system":"system_{sys_id}","allocations":"{wh_str}","token":"{secret}"}}')

    # ── Strategy equity curve (simulated) ───────────────────────────────────────
    # Simulated strategy equity compounding the selected assets' daily returns.
    # Derived only from immutable daily closes + the same signals used live, so it
    # is fully reproducible across restarts.
    try:
        from . import backtest
        sim = backtest.run_equity_simulation(
            s.all_assets, closes[s.all_assets], best, second, equal_weight,
            alloc_df, s.bench_trend, gold_trend,
            config.get("start_date", "2023-01-01"),
            bool(config.get("enable_total_filter", True)),
            bool(config.get("allow_gold_when_filtered", True)),
        )
        strategy_curve = [
            {
                "date":   d.date().isoformat(),
                "equity": round(float(eqv), 6),
                "asset":  clean_name(str(best.loc[d])),
            }
            for d, eqv in sim["equity"].items()
        ]
    except Exception as _e:
        print(f"  ! strategy equity curve build failed: {_e}")
        strategy_curve = None

    return {
        "date":           last_date.date().isoformat(),
        "allocation":     alloc_display,
        "webhook":        wh_str,
        "payload":        payload,
        "scores":         scores,
        "strategy_curve": strategy_curve,
    }


def main(config: dict = CONFIG) -> None:
    compute_signal(config)


if __name__ == "__main__":
    main()
