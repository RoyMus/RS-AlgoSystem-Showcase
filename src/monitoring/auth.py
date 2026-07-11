"""Single-password authentication for the FastAPI dashboard.

Responsibilities:
  - Issue and verify HMAC-signed session cookies (stdlib only: hmac, hashlib,
    base64, time). No third-party JWT library is used so there is no extra dep.
  - Constant-time password comparison to prevent timing-oracle attacks.
  - Per-IP brute-force lockout tracked in a module-level dict (safe for a
    single-process asyncio server).

NOTE: when auth_enabled() returns False the dashboard is served OPEN.
The caller (main.py) MUST log a warning in that case so operators notice
during local dev or when secrets are accidentally unset in production.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SESSION_COOKIE = "session"
SESSION_TTL = 7 * 24 * 3600  # ~7 days, in seconds

# ---------------------------------------------------------------------------
# Brute-force lockout constants and state
# ---------------------------------------------------------------------------

_MAX_FAILURES = 5
_LOCKOUT_SECONDS = 15 * 60  # 15 minutes

# Dict[ip_str, {"fails": int, "locked_until": float}]
_ip_state: dict[str, dict] = {}
_ip_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def auth_enabled() -> bool:
    """Return True only when BOTH SECRET_KEY and DASHBOARD_PASSWORD are set.

    We read env on every call so that test fixtures and fly.io secret rotations
    are picked up without restarting the process.  When False the dashboard is
    served open ONLY where open_access_allowed() permits it; otherwise the caller
    must fail closed.
    """
    return bool(os.environ.get("SECRET_KEY")) and bool(os.environ.get("DASHBOARD_PASSWORD"))


def open_access_allowed() -> bool:
    """Whether the dashboard may be served WITHOUT auth when secrets are unset.

    Fail-open is a local-dev convenience and must never happen in production:
      - On fly.io (FLY_APP_NAME / FLY_MACHINE_ID present in the env) it is NEVER
        allowed — a misconfigured/unset secret must fail closed, not silently
        expose the portfolio to the internet.
      - Anywhere else (local) it is allowed only when ALLOW_OPEN_DASHBOARD is
        explicitly truthy, so even local defaults to closed.

    Irrelevant when auth_enabled() is True (real auth runs in that case).
    """
    if os.environ.get("FLY_APP_NAME") or os.environ.get("FLY_MACHINE_ID"):
        return False
    val = (os.environ.get("ALLOW_OPEN_DASHBOARD") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Session token helpers
# ---------------------------------------------------------------------------


def _signing_key() -> bytes:
    """Return the SECRET_KEY bytes used to sign tokens.

    Callers must only invoke this when auth_enabled() is True so the env var
    is guaranteed to be non-empty.
    """
    return os.environ["SECRET_KEY"].encode()


def issue_session() -> str:
    """Return a signed session token valid for SESSION_TTL seconds.

    Format: ``<exp>.<sig>`` where ``exp`` is a Unix-epoch int and ``sig`` is
    a URL-safe base64-encoded HMAC-SHA256 over the string representation of
    ``exp`` (padding stripped so the token is URL-clean).
    """
    exp = int(time.time()) + SESSION_TTL
    sig = _compute_sig(exp)
    return f"{exp}.{sig}"


def verify_session(token: str | None) -> bool:
    """Return True when *token* is structurally valid, not expired, and the
    HMAC matches.  Any parse error or missing env var returns False silently
    so callers never have to deal with exceptions on the auth path.
    """
    if not token:
        return False
    if not auth_enabled():
        return False
    try:
        raw_exp, raw_sig = token.split(".", 1)
        exp = int(raw_exp)
    except (ValueError, AttributeError):
        return False
    if exp <= time.time():
        return False
    expected = _compute_sig(exp)
    try:
        return hmac.compare_digest(expected, raw_sig)
    except (TypeError, ValueError):
        return False


def _compute_sig(exp: int) -> str:
    """Compute the URL-safe base64 HMAC-SHA256 signature over str(exp).

    Padding is stripped with rstrip("=") so the resulting token contains no
    characters that need URL-encoding.
    """
    digest = hmac.new(_signing_key(), str(exp).encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


# ---------------------------------------------------------------------------
# Password check
# ---------------------------------------------------------------------------


def check_password(candidate: str | None) -> bool:
    """Constant-time comparison of *candidate* against DASHBOARD_PASSWORD.

    Returns False when the candidate is None, the env var is unset, or the
    strings differ.  Using hmac.compare_digest avoids timing side-channels that
    could allow an attacker to enumerate characters of the correct password.
    """
    if candidate is None:
        return False
    expected = os.environ.get("DASHBOARD_PASSWORD")
    if not expected:
        return False
    try:
        return hmac.compare_digest(expected, candidate)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Brute-force lockout
# ---------------------------------------------------------------------------


def check_locked(ip: str) -> float | None:
    """Return remaining lock seconds (>0) if the IP is currently locked, else None.

    Expired locks are cleared lazily on this call so the dict does not grow
    unboundedly for addresses that were locked long ago.
    """
    now = time.time()
    with _ip_lock:
        state = _ip_state.get(ip)
        if state is None:
            return None
        locked_until = state.get("locked_until", 0.0)
        if locked_until and locked_until > now:
            return locked_until - now
        # Lock has expired; clean it up.
        if locked_until and locked_until <= now:
            _ip_state.pop(ip, None)
        return None


def record_failure(ip: str) -> bool:
    """Increment the failure counter for *ip*.

    When the count reaches _MAX_FAILURES the IP is locked for _LOCKOUT_SECONDS,
    the counter is reset, and True is returned so callers can send an alert.
    Returns False when the threshold has not yet been reached.
    """
    now = time.time()
    with _ip_lock:
        state = _ip_state.setdefault(ip, {"fails": 0, "locked_until": 0.0})
        state["fails"] += 1
        if state["fails"] >= _MAX_FAILURES:
            state["locked_until"] = now + _LOCKOUT_SECONDS
            state["fails"] = 0
            return True
        return False


def reset_failures(ip: str) -> None:
    """Clear the failure counter and any active lock for *ip* on successful login.

    Called after a correct password so an address is never locked out of its
    own session after a typo-then-correct sequence.
    """
    with _ip_lock:
        _ip_state.pop(ip, None)
