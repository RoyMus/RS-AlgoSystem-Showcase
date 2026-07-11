from __future__ import annotations

import asyncio
import csv
import datetime
import hmac
import io
import logging
import logging.config
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

import os
import urllib.parse

from .config import load_config, resolve_config_path
from .execution.manager import ExchangeManager
from .models import OrderResult, TargetPortfolio, WebhookSignal
from .monitoring import auth
from .monitoring import cost_basis
from .monitoring import equity as equity_store
from .monitoring import strategy_equity
from .monitoring.commands import TelegramCommandListener
from .monitoring.equity import EquitySampler
from .monitoring.notifier import TelegramLogHandler, TelegramNotifier
from .monitoring.position_monitor import PositionMonitor
from .monitoring.reporter import WeeklyReporter
from .signals.scheduler import SignalScheduler
from .signals.processor import SignalProcessor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default"},
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/server.log",
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "formatter": "default",
                "encoding": "utf-8",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (populated in lifespan)
# ---------------------------------------------------------------------------
_config = None
_processor: SignalProcessor | None = None
_manager: ExchangeManager | None = None
_generator: SignalScheduler | None = None
_notifier: TelegramNotifier | None = None
_sampler: EquitySampler | None = None
_reporter: WeeklyReporter | None = None
_position_monitor: PositionMonitor | None = None
_command_listener: TelegramCommandListener | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _processor, _manager, _generator, _notifier, _sampler, _reporter
    global _position_monitor, _command_listener

    logger.info("Starting AutomateCryptoSignals server…")
    _config = load_config()
    logger.info("Loaded config from %s", resolve_config_path())
    _manager = ExchangeManager(_config)

    # Telegram notifier + instant error alerts: attach a handler to the root logger so
    # every existing logger.error(...) (failed orders, signal timeouts, price/symbol
    # not found, balance/earn errors) is pushed to Telegram, throttled per-message.
    _notifier = TelegramNotifier(
        bot_token=_config.notifications.telegram.bot_token,
        chat_id=_config.notifications.telegram.chat_id,
        throttle_seconds=_config.notifications.throttle_seconds,
    )
    logging.getLogger().addHandler(TelegramLogHandler(_notifier))

    # Bootstrap positions from the exchange on every startup so state/positions.json
    # always reflects reality even before the first signal arrives, and seed/clean the
    # cost-basis ledger against what is actually held.
    await _manager.snapshot_positions()
    await _reconcile_cost_basis()

    async def _rebalance_and_snapshot(target: TargetPortfolio) -> List[OrderResult]:
        results = await _manager.execute_rebalance(target)
        # Fold real fills into the cost-basis ledger before re-snapshotting, then
        # reconcile/seed against the fresh holdings.
        cost_basis.apply_orders(results)
        await _manager.snapshot_positions()
        await _reconcile_cost_basis()
        _notify_rebalance(target, results)
        return results

    _processor = SignalProcessor(_config)
    _processor.start(on_target=_rebalance_and_snapshot)

    _generator = SignalScheduler(_config.signal_generator, _processor)
    _generator.start()

    # Equity time-series sampler (powers the dashboard curve + weekly P&L) and the
    # weekly Telegram report.
    _sampler = EquitySampler(_manager, _config.reporting.equity_sample_interval_minutes)
    _sampler.start()
    _reporter = WeeklyReporter(_config.reporting, _manager, _notifier)
    _reporter.start()

    # Profit monitor (sends "X is up Y%, trim it?" prompts) + inbound command listener
    # (handles the `rebalance` / `yes` chat replies). Both share the existing notifier.
    _position_monitor = PositionMonitor(_config.position_monitor, _manager, _processor, _notifier)
    _position_monitor.start()
    _command_listener = TelegramCommandListener(
        bot_token=_config.notifications.telegram.bot_token,
        chat_id=_config.notifications.telegram.chat_id,
        loop=asyncio.get_running_loop(),
        notifier=_notifier,
        processor=_processor,
        manager=_manager,
        execute_target=_rebalance_and_snapshot,
    )
    _command_listener.start()

    logger.info("Active clients: %s", _manager.client_labels)
    logger.info("Registered signal systems: %s", list(_config.signals.keys()))
    _notifier.notify("✅ AutomateCryptoSignals started.")

    yield  # ← server is running

    logger.info("Shutting down…")
    if _command_listener is not None:
        _command_listener.stop()
    if _position_monitor is not None:
        await _position_monitor.stop()
    if _reporter is not None:
        await _reporter.stop()
    if _sampler is not None:
        await _sampler.stop()
    if _generator is not None:
        await _generator.stop()
    await _processor.stop()
    await _manager.close()


app = FastAPI(
    title="AutomateCryptoSignals",
    description="Webhook-driven multi-exchange crypto trade execution server",
    version="1.0.0",
    lifespan=lifespan,
    # Disable the auto-generated, unauthenticated API docs. On a public money app
    # these would publish the full endpoint list + request schemas (including the
    # admin routes) to anyone. None on all three removes /docs, /redoc and
    # /openapi.json entirely.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_matches(expected: Optional[str], candidates: List[str]) -> bool:
    """Constant-time check that one of *candidates* equals *expected*.

    Uses hmac.compare_digest per candidate so a network attacker can't recover the
    admin token byte-by-byte via response-timing differences (a plain ``in``/``==``
    short-circuits on the first differing byte). Returns False when no token is
    configured or on any type/encoding error.
    """
    if not expected:
        return False
    for cand in candidates:
        try:
            if hmac.compare_digest(expected, cand):
                return True
        except (TypeError, ValueError):
            continue
    return False


async def require_admin_token(
    authorization: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> None:
    """Guard mutating/sensitive endpoints once the app is publicly reachable.

    Accepts the configured server.webhook_token via 'Authorization: Bearer <t>',
    the 'X-Token' header, or a '?token=' query parameter. Fails closed: if no token
    is configured at all, these endpoints are disabled rather than left open.
    Only /dashboard, /public/* and /health stay open.
    """
    expected = _config.server.webhook_token if _config else None
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin token not configured — guarded endpoints are disabled",
        )
    candidates: List[str] = []
    if authorization and authorization.lower().startswith("bearer "):
        candidates.append(authorization[7:].strip())
    if x_token:
        candidates.append(x_token)
    if token:
        candidates.append(token)
    if not _token_matches(expected, candidates):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing admin token")


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction that works behind fly.io's proxy.

    Preference order:
      1. ``Fly-Client-IP`` — set by fly.io's edge, most reliable on that platform.
      2. First entry of ``X-Forwarded-For`` — standard proxy header (may be spoofed
         when not behind a trusted proxy, but acceptable here for rate-limiting).
      3. ``request.client.host`` — the direct TCP peer, always available locally.
    Falls back to the string ``"unknown"`` so callers never receive None.
    """
    fly_ip = request.headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _auth_unconfigured_response() -> None:
    """Raised by dashboard guards when auth is unset and open access is NOT allowed.

    Fails closed (503) so production never serves portfolio data unauthenticated.
    Open access is only permitted locally via ALLOW_OPEN_DASHBOARD (see
    auth.open_access_allowed)."""
    if auth.open_access_allowed():
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Dashboard authentication not configured — set SECRET_KEY and DASHBOARD_PASSWORD",
    )


async def require_dashboard_session(request: Request) -> None:
    """Gate the dashboard data feeds behind a valid signed session cookie.

    When auth is enabled, an invalid/missing cookie raises 401 so the browser's
    fetch() calls redirect to /login. When auth is UNCONFIGURED it fails closed
    (503) in production and only serves open where open_access_allowed() permits.
    """
    if not auth.auth_enabled():
        return _auth_unconfigured_response()
    if not auth.verify_session(request.cookies.get(auth.SESSION_COOKIE)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


async def require_dashboard_or_token(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> None:
    """Allow access via a valid dashboard session cookie OR the admin token.
    Lets the browser (cookie) and control.py (token) both read ASE data. The
    admin token works regardless of dashboard auth; open (no-auth) access is only
    served where open_access_allowed() permits, otherwise it fails closed."""
    if auth.auth_enabled() and auth.verify_session(request.cookies.get(auth.SESSION_COOKIE)):
        return
    expected = _config.server.webhook_token if _config else None
    candidates = []
    if authorization and authorization.lower().startswith("bearer "):
        candidates.append(authorization[7:].strip())
    if x_token:
        candidates.append(x_token)
    if token:
        candidates.append(token)
    if _token_matches(expected, candidates):
        return
    if not auth.auth_enabled():
        return _auth_unconfigured_response()
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


async def _reconcile_cost_basis() -> None:
    """Clamp the cost-basis ledger to live holdings and adopt any unledgered position.

    Uses a fresh equity valuation (merged positions with amounts) — cheap relative to a
    rebalance and keeps the ledger honest after every snapshot.
    """
    if _manager is None:
        return
    try:
        snapshot = await _manager.equity_snapshot()
        cost_basis.reconcile(snapshot)
        cost_basis.seed_missing(snapshot)
    except Exception as exc:  # noqa: BLE001 — never block trading on ledger upkeep
        logger.warning("Cost-basis reconcile failed: %s", exc)


def _notify_rebalance(target: "TargetPortfolio", results: List[OrderResult]) -> None:
    """Push a 'signal executed' confirmation to Telegram after a rebalance.

    Fires only when orders were actually placed (skips no-op rebalances that were
    already at target). Each order line is marked ✅/❌; failures still also surface
    via the ERROR-log alert path, but this gives the full execution picture at a glance.
    """
    if _notifier is None or not _notifier.enabled or not results:
        return

    tgt = "  ".join(f"{sym.split('/')[0]} {frac*100:.0f}%" for sym, frac in target.targets.items()) or "CASH"
    failed = [r for r in results if r.status == "error"]
    header = "⚠️ <b>Rebalance executed with errors</b>" if failed else "✅ <b>Rebalance executed</b>"
    lines = [header, f"<b>Target:</b> {tgt}", ""]
    for r in results:
        mark = "❌" if r.status == "error" else "✅"
        line = f"{mark} {r.side.upper()} {r.symbol} {r.quantity:.6g} [{r.exchange}]"
        if r.status == "error" and r.error:
            line += f" — {r.error[:120]}"
        lines.append(line)
    _notifier.notify("\n".join(lines))


# Dashboard HTML is bundled with the package and read once at startup.
_DASHBOARD_HTML = (Path(__file__).parent / "monitoring" / "dashboard.html").read_text(encoding="utf-8")

# Login page — created by a separate agent.  Fall back to a minimal inline form
# so the module stays importable even before that file exists.
_LOGIN_HTML_PATH = Path(__file__).parent / "monitoring" / "login.html"
try:
    _LOGIN_HTML = _LOGIN_HTML_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning(
        "login.html not found at %s — using inline fallback form. "
        "Deploy the real login page to remove this warning.",
        _LOGIN_HTML_PATH,
    )
    _LOGIN_HTML = (
        '<form method="POST" action="/login">'
        '<input type="password" name="password">'
        "<button>Log in</button>"
        "</form>"
    )

# Cap the number of points sent to the browser so a long history stays light.
_MAX_DASHBOARD_POINTS = 600


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "exchanges": _manager.client_labels if _manager else [],
        "signals": list(_config.signals.keys()) if _config else [],
    }


@app.post("/state", summary="Overwrite signal system states and reload in-memory state", dependencies=[Depends(require_admin_token)])
async def set_state(request: Request) -> Dict[str, Any]:
    if _processor is None:
        raise HTTPException(status_code=503)
    body = await request.body()
    try:
        import json as _json
        raw = _json.loads(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")
    loaded = _processor.set_state(raw)
    return {"status": "ok", "loaded_systems": loaded}


@app.post("/test/rebalance", summary="Simulate a rebalance using real balances but place no orders", dependencies=[Depends(require_admin_token)])
async def test_rebalance(request: Request) -> Dict[str, Any]:
    if _manager is None or _processor is None:
        raise HTTPException(status_code=503)

    body = await request.body()
    signals = None
    if body.strip():
        try:
            import json as _json
            raw = _json.loads(body)
            entries = raw if isinstance(raw, list) else [raw]
            signals = [WebhookSignal.model_validate(e) for e in entries]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    portfolio = _processor.preview_portfolio(signals)
    if portfolio is None:
        raise HTTPException(status_code=400, detail="No signals in current state — send a signal body or set state first")

    results = await _manager.simulate_rebalance(portfolio)
    return {
        "portfolio": {
            "targets": {sym: round(frac, 4) for sym, frac in portfolio.targets.items()},
            "quote": portfolio.quote,
            "protected_symbols": sorted(portfolio.protected_symbols),
            "protected_fraction": {k: round(v, 4) for k, v in portfolio.protected_fraction.items()},
        },
        "orders": [r.model_dump() for r in results],
    }


@app.post("/signal/run", summary="Run the built-in RS Algo signal generator now (computes, executes only if changed)", dependencies=[Depends(require_admin_token)])
async def run_signal() -> Dict[str, Any]:
    if _generator is None:
        raise HTTPException(status_code=503)
    if not _config.signal_generator.enabled:
        raise HTTPException(status_code=400, detail="signal_generator is disabled in config.yaml")
    results = await _generator.run_once()
    if not results:
        raise HTTPException(status_code=502, detail="All signal instances failed or timed out — see logs")
    return {
        "instances": [
            {
                "system": r["system"],
                "date": r["date"],
                "allocation": r["allocation"],
                "webhook": r["webhook"],
                "scores": r["scores"],
            }
            for r in results
        ]
    }


@app.get("/state", summary="Current in-memory signal system states", dependencies=[Depends(require_admin_token)])
async def get_state() -> Dict[str, Any]:
    if _processor is None:
        raise HTTPException(status_code=503)
    systems = {}
    for name, st in _processor._system_states.items():
        systems[name] = {
            "signal": [{"symbol": a.symbol, "allocation": a.allocation} for a in st.signal],
            "owned_symbols": sorted(st.owned_symbols),
        }
    return {"systems": systems}


# ---------------------------------------------------------------------------
# Public, read-only monitoring (no token) — dashboard + its data feeds
# ---------------------------------------------------------------------------


@app.get("/dashboard", response_class=HTMLResponse, response_model=None, summary="Equity-curve & allocation dashboard")
async def dashboard(request: Request) -> HTMLResponse | RedirectResponse:
    """Serve the dashboard HTML, or redirect to /login when auth is enabled and
    the session cookie is absent or invalid. When auth is UNCONFIGURED the page is
    served open only where open_access_allowed() permits (local dev); in production
    it fails closed (503) so the portfolio is never exposed unauthenticated."""
    if auth.auth_enabled():
        if not auth.verify_session(request.cookies.get(auth.SESSION_COOKIE)):
            return RedirectResponse("/login", status_code=303)
        return HTMLResponse(content=_DASHBOARD_HTML)
    if not auth.open_access_allowed():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard authentication not configured — set SECRET_KEY and DASHBOARD_PASSWORD",
        )
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/login", response_class=HTMLResponse, response_model=None, summary="Dashboard login page")
async def login_page(request: Request) -> HTMLResponse | RedirectResponse:
    """Serve the login form, or redirect to /dashboard when the session is already valid.

    Skips the redirect when auth is disabled so /login always shows the form in
    environments where auth is intentionally off (avoids an infinite redirect).
    """
    if auth.auth_enabled() and auth.verify_session(request.cookies.get(auth.SESSION_COOKIE)):
        return RedirectResponse("/dashboard", status_code=303)
    return HTMLResponse(content=_LOGIN_HTML)


@app.post("/login", summary="Validate password and issue a session cookie")
async def login(request: Request) -> RedirectResponse:
    """Authenticate with the single dashboard password.

    On success: issue a signed session cookie and redirect to /dashboard.
    On failure: record the attempt for brute-force tracking and redirect back
    to /login?error=1 so the form can display an error message.
    A locked IP receives 429 with a Retry-After header instead of the form.
    """
    ip = _client_ip(request)

    remaining = auth.check_locked(ip)
    if remaining is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {int(remaining)} seconds.",
            headers={"Retry-After": str(int(remaining))},
        )

    body = await request.body()
    password = urllib.parse.parse_qs(body.decode()).get("password", [None])[0]

    if auth.check_password(password):
        auth.reset_failures(ip)
        resp = RedirectResponse("/dashboard", status_code=303)
        resp.set_cookie(
            auth.SESSION_COOKIE,
            auth.issue_session(),
            max_age=auth.SESSION_TTL,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
        )
        return resp

    locked_now = auth.record_failure(ip)
    if locked_now and _notifier and _notifier.enabled:
        _notifier.notify(f"🔒 Dashboard login locked for IP {ip} after repeated failures.")
    return RedirectResponse("/login?error=1", status_code=303)


@app.post("/logout", summary="Clear the session cookie and redirect to login")
async def logout() -> RedirectResponse:
    """Invalidate the dashboard session by deleting the cookie client-side."""
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    return resp


def _dominant_asset(rec: Dict[str, Any]) -> str:
    """The largest-by-value holding in a sample → used to colour the curve segment.
    Falls back to CASH when there are no token positions."""
    positions = rec.get("positions") or []
    if not positions:
        return "CASH"
    top = max(positions, key=lambda p: p.get("value") or 0.0)
    if (top.get("value") or 0.0) <= 0.0:
        return "CASH"
    return top.get("base") or "CASH"


@app.get("/public/equity", summary="Equity curve points for the dashboard", dependencies=[Depends(require_dashboard_session)])
async def public_equity() -> Dict[str, Any]:
    full = equity_store.read_history()
    # Deposit-adjusted (TWR) curve: deposits/withdrawals are divided out so funding the
    # account doesn't read as a gain. Same dollar units, anchored at the first sample.
    adj = equity_store.twr_adjust(
        [float(r.get("total_value") or 0.0) for r in full],
        [float(r.get("flow") or 0.0) for r in full],
    )
    # Max drawdown over the FULL adjusted series (before downsampling, so it's exact).
    peak = 0.0
    max_dd = 0.0
    for v in adj:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    points = [
        {"ts": r.get("ts"), "total_value": v, "asset": _dominant_asset(r)}
        for r, v in zip(full, adj)
    ]
    # Downsample evenly to keep the payload light on long histories.
    if len(points) > _MAX_DASHBOARD_POINTS:
        step = len(points) / _MAX_DASHBOARD_POINTS
        points = [points[int(i * step)] for i in range(_MAX_DASHBOARD_POINTS)] + [points[-1]]
    quote = full[-1].get("quote") if full else None
    return {
        "quote": quote,
        "current": full[-1].get("total_value") if full else None,  # real balance (incl. deposits)
        "max_dd": round(max_dd, 4),
        "points": points,
    }


@app.get("/public/strategy-equity", summary="Simulated strategy equity curve (base_eq) for the dashboard", dependencies=[Depends(require_dashboard_session)])
async def public_strategy_equity() -> Dict[str, Any]:
    """Anchored, normalised strategy-equity curve (mirrors the .pine `base_eq`).

    Recomputed deterministically by the signal generator each cycle; this route
    just serves the cached payload. Empty until the generator has run once in a
    mode that produces a curve.
    """
    payload = strategy_equity.read_payload()
    if payload is None:
        return {"anchor": None, "current": None, "max_dd": None, "points": []}
    return payload


@app.get("/public/positions", summary="Current portfolio allocation for the dashboard", dependencies=[Depends(require_dashboard_session)])
async def public_positions() -> Dict[str, Any]:
    latest = equity_store.read_latest()
    if latest is None:
        return {"timestamp": None, "quote": None, "total_value": None,
                "quote_free": 0.0, "earn_value": 0.0, "positions": []}
    return {
        "timestamp": latest.get("timestamp"),
        "quote": latest.get("quote"),
        "total_value": latest.get("total_value"),
        "quote_free": latest.get("quote_free", 0.0),
        "earn_value": latest.get("earn_value", 0.0),
        "positions": latest.get("positions", []),
    }


@app.get("/public/profit", summary="Per-position unrealized P&L vs cost basis", dependencies=[Depends(require_dashboard_or_token)])
async def public_profit() -> Dict[str, Any]:
    """Per-position unrealized P&L using the cost-basis ledger.

    Takes a fresh valuation so prices/profit are current, reconciling the ledger to live
    holdings first. profit_pct is null for positions whose basis isn't known yet.
    """
    if _manager is None:
        raise HTTPException(status_code=503)
    snapshot = await _manager.equity_snapshot()
    cost_basis.reconcile(snapshot)
    cost_basis.seed_missing(snapshot)
    return {"quote": snapshot.get("quote"), "positions": cost_basis.profit_table(snapshot)}


@app.get("/public/accounts", summary="Per-account equity & P&L breakdown for the dashboard", dependencies=[Depends(require_dashboard_session)])
async def public_accounts() -> Dict[str, Any]:
    """Per-account (per-exchange-client) value series + current holdings.

    Time-series values come from the per-account breakdown now recorded in
    equity_history.jsonl (so each account's actual P&L can be tracked when
    several accounts run the same strategy). Current value + live positions come
    from the latest full snapshot. Accounts appear once they've been sampled.
    """
    full = equity_store.read_history()
    latest = equity_store.read_latest() or {}
    latest_clients: Dict[str, Any] = latest.get("clients") or {}

    # Stable account ordering: history order first, then any new ones from latest.
    names: List[str] = []
    for rec in full:
        for n in (rec.get("accounts") or {}):
            if n not in names:
                names.append(n)
    for n in latest_clients:
        if n not in names:
            names.append(n)

    quote = (full[-1].get("quote") if full else None) or latest.get("quote")

    accounts: List[Dict[str, Any]] = []
    for name in names:
        # Build full series (un-downsampled) with asset label per point. Values are
        # deposit-adjusted (TWR) so this account's funding isn't counted as P&L —
        # this is what made the Bybit curve appear to gain its entire account size.
        raw: List[Dict[str, Any]] = []
        for rec in full:
            acc = (rec.get("accounts") or {}).get(name)
            if not acc:
                continue
            asset = acc.get("asset") or _dominant_asset(rec)
            raw.append({"ts": rec.get("ts"), "value": acc.get("value"),
                        "flow": acc.get("flow") or 0.0, "asset": asset})
        adj = equity_store.twr_adjust(
            [float(p["value"] or 0.0) for p in raw],
            [float(p["flow"]) for p in raw],
        )
        full_series: List[Dict[str, Any]] = []
        peak = 0.0
        max_dd = 0.0
        for p, v in zip(raw, adj):
            full_series.append({"ts": p["ts"], "value": v, "asset": p["asset"]})
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)

        series = full_series
        if len(series) > _MAX_DASHBOARD_POINTS:
            step = len(series) / _MAX_DASHBOARD_POINTS
            series = [series[int(i * step)] for i in range(_MAX_DASHBOARD_POINTS)] + [series[-1]]

        cli = latest_clients.get(name) or {}
        current = cli.get("total_value")
        if current is None and series:
            current = series[-1].get("value")
        # Exposure: prefer the latest snapshot value (already written by the sampler);
        # fall back to 1.0 for accounts with no exposure field in the snapshot.
        exposure = round(float(cli.get("exposure", 1.0)), 4)
        accounts.append({
            "name": name,
            "current": current,
            "quote_free": cli.get("quote_free", 0.0),
            "earn_value": cli.get("earn_value", 0.0),
            "max_dd": round(max_dd, 4),
            "exposure": exposure,
            "points": series,
            "positions": cli.get("positions", []),
        })

    return {"quote": quote, "accounts": accounts}


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_\-]")


@app.get("/public/accounts/export", summary="Export per-account (or combined) equity history as CSV", dependencies=[Depends(require_dashboard_session)])
async def public_accounts_export(account: str = Query(..., description="Account name or '__combined__'")) -> PlainTextResponse:
    """Download the full (un-downsampled) equity history for one account or all accounts
    combined as a CSV with columns: timestamp, equity, since_start_abs, since_start_pct.
    """
    full = equity_store.read_history()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "equity", "since_start_abs", "since_start_pct"])

    # Exported equity is deposit-adjusted (TWR), matching the dashboard curve.
    if account == "__combined__":
        # For each timestamp sum the value (and flow) of every account in that record.
        ts_list: List[str] = []
        vals: List[float] = []
        flows: List[float] = []
        for rec in full:
            accs = rec.get("accounts") or {}
            if not accs:
                continue
            ts_list.append(rec.get("ts"))
            vals.append(sum(float(a.get("value") or 0.0) for a in accs.values()))
            flows.append(sum(float(a.get("flow") or 0.0) for a in accs.values()))
    else:
        ts_list = []
        vals = []
        flows = []
        for rec in full:
            acc = (rec.get("accounts") or {}).get(account)
            if acc is None:
                continue
            ts_list.append(rec.get("ts"))
            vals.append(float(acc.get("value") or 0.0))
            flows.append(float(acc.get("flow") or 0.0))

    adj = equity_store.twr_adjust(vals, flows)
    first_eq = adj[0] if adj else None
    for ts, eq in zip(ts_list, adj):
        abs_delta = round(eq - first_eq, 4) if first_eq is not None else ""
        pct_delta = round((eq - first_eq) / first_eq, 6) if first_eq else ""
        writer.writerow([ts, round(eq, 4), abs_delta, pct_delta])

    safe_name = _SAFE_FILENAME_RE.sub("_", account)
    filename = f"{safe_name}_equity.csv"
    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/public/signal", summary="Current signal targets from the strategy processor", dependencies=[Depends(require_dashboard_session)])
async def public_signal() -> Dict[str, Any]:
    """Returns the current strategy signal: targets list + quote currency.

    Safe to call at any time: returns empty/cash gracefully when no processor is ready
    or when the strategy is fully in cash. Never raises a 500.
    """
    try:
        if _processor is None:
            return {"targets": [], "quote": None, "cash": True}
        target = _processor.preview_portfolio()
        if target is None or not target.targets:
            quote = target.quote if target is not None else None
            return {"targets": [], "quote": quote, "cash": True}
        return {
            "targets": [
                {"base": sym.split("/")[0], "frac": frac}
                for sym, frac in target.targets.items()
            ],
            "quote": target.quote,
            "cash": False,
        }
    except Exception:
        logger.exception("Error in /public/signal")
        return {"targets": [], "quote": None, "cash": True}


@app.post("/report/now", summary="Send the weekly performance report immediately (for testing)", dependencies=[Depends(require_admin_token)])
async def report_now() -> Dict[str, Any]:
    if _reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not ready")
    text = await _reporter.send_report()
    return {"status": "sent" if _notifier and _notifier.enabled else "built (telegram disabled)", "message": text}


@app.post("/admin/equity/reset", summary="Clear equity history to re-baseline the curve (no trades)", dependencies=[Depends(require_admin_token)])
async def reset_equity() -> Dict[str, Any]:
    removed = equity_store.reset_history()
    snap = await _sampler.sample_once() if _sampler is not None else None
    return {
        "status": "ok",
        "removed_points": removed,
        "baseline": snap.get("total_value") if snap else None,
        "quote": snap.get("quote") if snap else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        # Bound graceful shutdown so Ctrl+C / SIGTERM exits promptly instead of
        # hanging on background tasks (sampler, reporter, Telegram long-poll) and
        # leaving the port bound — which is what orphans a stale server on restart.
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
