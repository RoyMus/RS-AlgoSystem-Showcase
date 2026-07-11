from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")

# Where the live, instance-owned config lives on the fly.io persistent volume.
# This file is NOT in git and NOT baked into the image — it is edited on the
# running instance over fly's authenticated SSH (fly ssh console / sftp) so config
# changes never require a code push/redeploy and are never exposed on GitHub.
_VOLUME_CONFIG = Path("/app/state/config.yaml")
# Sanitised template shipped in the image, used to seed the volume on first boot.
_TEMPLATE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"


def resolve_config_path() -> Path:
    """Resolve which config file to load, in priority order:

      1. $CONFIG_PATH               — explicit override (set on fly.io)
      2. /app/state/config.yaml     — the instance config on the persistent volume
      3. ./config.yaml              — local-dev fallback

    This is what makes config "owned by the instance": on fly we point CONFIG_PATH
    at the volume so deploys never carry config, while local dev keeps using the
    repo-root config.yaml (which is gitignored).
    """
    env = os.getenv("CONFIG_PATH")
    if env:
        return Path(env)
    if _VOLUME_CONFIG.exists():
        return _VOLUME_CONFIG
    return Path("config.yaml")


def seed_config_if_missing(path: Path) -> None:
    """On first boot the volume is empty — copy the bundled template into place so
    there is a valid config to edit. Never overwrites an existing config."""
    if path.exists():
        return
    if not _TEMPLATE_CONFIG.exists():
        logger.warning("No config at %s and no template at %s to seed from.", path, _TEMPLATE_CONFIG)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Seeded config from template %s → %s", _TEMPLATE_CONFIG, path)


def _resolve_env(value: str) -> str:
    """Substitute ${VAR} placeholders with environment variable values."""
    def _sub(match: re.Match) -> str:
        var = match.group(1)
        resolved = os.getenv(var)
        if resolved is None:
            raise EnvironmentError(f"Required environment variable '{var}' is not set")
        return resolved

    return _ENV_PATTERN.sub(_sub, value) if isinstance(value, str) else value


@dataclass
class SignalConfig:
    weight: float  # 0.0 – 1.0


@dataclass
class ClientConfig:
    """Credentials for a single trading account on an exchange."""
    api_key: str
    api_secret: str
    testnet: bool = False
    passphrase: Optional[str] = None        # Required by some exchanges (e.g. OKX)
    label: Optional[str] = None             # Human-readable name used in logs
    dry_run: bool = False                   # Fetch balance+price but skip create_order
    demo: bool = False                      # Use exchange's demo/paper trading mode
    use_earn: bool = False                  # Kraken only: deposit idle cash into flexible earn
    portfolio_exposure: Optional[float] = None  # Overrides global if set (0.0–1.0)
    market_type: str = "spot"               # "spot" or "futures" (per client)
    leverage: int = 1                       # Leverage multiplier for futures (1 = unlevered)
    margin_mode: str = "isolated"           # "isolated" or "cross" (futures only)


@dataclass
class ExchangeConfig:
    enabled: bool
    clients: List[ClientConfig]       # One entry per trading account
    quote_currency: str = "USDT"      # Quote currency used for all pairs on this exchange


@dataclass
class ExecutionConfig:
    aggregation_window_ms: int = 200  # How long to wait before flushing the batch
    min_allocation: float = 0.001     # Net allocations below this fraction are skipped (0.1%)
    min_trade_value: float = 5.0      # Skip sells/buys worth less than this in quote currency
    order_delay_ms: int = 1000        # Pause between consecutive orders (avoids nonce collisions)
    portfolio_exposure: float = 1.0   # Fraction of total portfolio to deploy (0.0–1.0)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    webhook_token: Optional[str] = None  # Static token checked via ?token= query param


@dataclass
class GeneratorSystemConfig:
    """One RS Algo instance: a signal `system` plus per-instance strategy overrides
    (e.g. its own `assets` basket) merged over signals.engine.CONFIG."""
    system: str                                  # which entry in `signals:` this maps to
    overrides: Dict[str, Any] = field(default_factory=dict)  # merged over signals.engine.CONFIG


@dataclass
class SignalGeneratorConfig:
    """Built-in RS Algo signal generator that runs in-process on a daily schedule.

    When enabled, the server computes each instance's target allocation itself
    (signals.engine.compute_signal) and feeds it straight to the executor — no
    external webhook is involved. Multiple instances (e.g. system_1 / system_2)
    can run with different baskets; they share one daily schedule.
    """
    enabled: bool = False
    run_at_utc: str = "00:10"     # daily run time (UTC), just after the Binance daily close
    run_on_start: bool = False    # also compute once on startup (handy for testing)
    timeout_seconds: int = 300    # abort a single instance's run that hangs longer than this
    systems: List[GeneratorSystemConfig] = field(default_factory=list)


@dataclass
class TelegramConfig:
    """Telegram bot credentials for reports and instant error alerts."""
    enabled: bool = False
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass
class NotificationsConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    throttle_seconds: int = 300   # suppress a repeated alert (same dedup key) for this long


@dataclass
class ReportingConfig:
    """Equity sampling + weekly performance report settings."""
    equity_sample_interval_minutes: int = 60
    weekly_report_day: str = "mon"        # mon..sun (UTC)
    weekly_report_at_utc: str = "08:00"   # HH:MM (UTC)
    # Account (exchange-client) labels to include in the weekly report, each shown as its
    # own section. Empty = include every account. e.g. ["kraken-main"].
    report_accounts: List[str] = field(default_factory=list)


@dataclass
class PositionMonitorConfig:
    """Profit monitor: alert on Telegram when a position is up past a threshold and
    offer to rebalance it. Requires notifications.telegram to be enabled to do anything.
    The inbound command listener (the `rebalance`/`yes` chat commands) is enabled
    whenever Telegram is enabled — it does not depend on this block."""
    enabled: bool = False
    profit_threshold_pct: float = 25.0    # alert when unrealized P&L on a position >= this
    check_interval_minutes: int = 60      # how often to evaluate positions
    alert_throttle_hours: int = 24        # don't re-alert the same base more often than this


@dataclass
class AppConfig:
    signals: Dict[str, SignalConfig] = field(default_factory=dict)
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    signal_generator: SignalGeneratorConfig = field(default_factory=SignalGeneratorConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    position_monitor: PositionMonitorConfig = field(default_factory=PositionMonitorConfig)

    @property
    def enabled_exchanges(self) -> Dict[str, ExchangeConfig]:
        return {name: cfg for name, cfg in self.exchanges.items() if cfg.enabled}


def load_config(path: str | Path | None = None) -> AppConfig:
    # No explicit path → resolve the instance config (and seed it on first boot).
    # An explicit path is passed only for validation of candidate config text.
    if path is None:
        path = resolve_config_path()
        seed_config_if_missing(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    load_dotenv(override=False)

    signals = {
        name: SignalConfig(weight=float(cfg["weight"]))
        for name, cfg in raw.get("signals", {}).items()
    }

    exchanges: Dict[str, ExchangeConfig] = {}
    for name, cfg in raw.get("exchanges", {}).items():
        # Exchange-level testnet default (can be overridden per client)
        default_testnet = bool(cfg.get("testnet", False))

        enabled = bool(cfg.get("enabled", True))

        clients: List[ClientConfig] = []
        if enabled:
            for idx, client_raw in enumerate(cfg.get("clients", [])):
                raw_exp = client_raw.get("portfolio_exposure")
                if raw_exp is not None:
                    raw_exp = float(raw_exp)
                    if not 0.0 < raw_exp <= 1.0:
                        raise ValueError(
                            f"portfolio_exposure for client '{client_raw.get('label', idx)}' "
                            f"must be between 0 (exclusive) and 1.0, got {raw_exp}"
                        )
                market_type = client_raw.get("market_type", "spot")
                if market_type not in ("spot", "futures"):
                    raise ValueError(
                        f"market_type for client '{client_raw.get('label', idx)}' "
                        f"must be 'spot' or 'futures', got '{market_type}'"
                    )
                leverage = int(client_raw.get("leverage", 1))
                if leverage < 1:
                    raise ValueError(
                        f"leverage for client '{client_raw.get('label', idx)}' must be >= 1"
                    )
                margin_mode = client_raw.get("margin_mode", "isolated")
                if margin_mode not in ("isolated", "cross"):
                    raise ValueError(
                        f"margin_mode for client '{client_raw.get('label', idx)}' "
                        f"must be 'isolated' or 'cross', got '{margin_mode}'"
                    )
                clients.append(ClientConfig(
                    api_key=_resolve_env(client_raw["api_key"]),
                    api_secret=_resolve_env(client_raw["api_secret"]),
                    testnet=bool(client_raw.get("testnet", default_testnet)),
                    passphrase=_resolve_env(client_raw["passphrase"]) if client_raw.get("passphrase") else None,
                    label=client_raw.get("label") or f"{name}_{idx + 1}",
                    dry_run=bool(client_raw.get("dry_run", False)),
                    demo=bool(client_raw.get("demo", False)),
                    portfolio_exposure=raw_exp,
                    market_type=market_type,
                    leverage=leverage,
                    margin_mode=margin_mode,
                    use_earn=bool(client_raw.get("use_earn", False)),
                ))

        exchanges[name] = ExchangeConfig(
            enabled=enabled,
            clients=clients,
            quote_currency=cfg.get("quote_currency", "USDT"),
        )

    exec_raw = raw.get("execution", {})
    exposure = float(exec_raw.get("portfolio_exposure", 1.0))
    if not 0.0 < exposure <= 1.0:
        raise ValueError(f"portfolio_exposure must be between 0 (exclusive) and 1.0, got {exposure}")
    execution = ExecutionConfig(
        aggregation_window_ms=int(exec_raw.get("aggregation_window_ms", 200)),
        min_allocation=float(exec_raw.get("min_allocation", 0.001)),
        min_trade_value=float(exec_raw.get("min_trade_value", 5.0)),
        order_delay_ms=int(exec_raw.get("order_delay_ms", 1000)),
        portfolio_exposure=exposure,
    )

    srv_raw = raw.get("server", {})
    raw_token = srv_raw.get("webhook_token")
    server = ServerConfig(
        host=srv_raw.get("host", "0.0.0.0"),
        port=int(srv_raw.get("port", 8000)),
        webhook_token=_resolve_env(raw_token) if raw_token else None,
    )

    sg_raw = raw.get("signal_generator", {})
    run_at = str(sg_raw.get("run_at_utc", "00:10"))
    try:
        hh, mm = (int(x) for x in run_at.split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError
    except Exception:
        raise ValueError(f"signal_generator.run_at_utc must be 'HH:MM' (24h UTC), got '{run_at}'")

    # Accept a `systems:` list, or a single `system:` (back-compat) → one instance.
    raw_systems = sg_raw.get("systems")
    if raw_systems is None and sg_raw.get("system"):
        raw_systems = [{"system": sg_raw["system"]}]
    gen_systems: List[GeneratorSystemConfig] = []
    for entry in (raw_systems or []):
        if "system" not in entry:
            raise ValueError("each signal_generator.systems entry needs a 'system' key")
        overrides = {k: v for k, v in entry.items() if k != "system"}
        gen_systems.append(GeneratorSystemConfig(system=str(entry["system"]), overrides=overrides))

    signal_generator = SignalGeneratorConfig(
        enabled=bool(sg_raw.get("enabled", False)),
        run_at_utc=run_at,
        run_on_start=bool(sg_raw.get("run_on_start", False)),
        timeout_seconds=int(sg_raw.get("timeout_seconds", 300)),
        systems=gen_systems,
    )

    notif_raw = raw.get("notifications", {}) or {}
    tg_raw = notif_raw.get("telegram", {}) or {}
    tg_enabled = bool(tg_raw.get("enabled", False))
    # Only resolve ${VAR} placeholders when Telegram is enabled, so the app can
    # run without the secrets set when notifications are off.
    telegram = TelegramConfig(
        enabled=tg_enabled,
        bot_token=_resolve_env(tg_raw["bot_token"]) if tg_enabled and tg_raw.get("bot_token") else None,
        chat_id=_resolve_env(tg_raw["chat_id"]) if tg_enabled and tg_raw.get("chat_id") else None,
    )
    notifications = NotificationsConfig(
        telegram=telegram,
        throttle_seconds=int(notif_raw.get("throttle_seconds", 300)),
    )

    rep_raw = raw.get("reporting", {}) or {}
    report_at = str(rep_raw.get("weekly_report_at_utc", "08:00"))
    try:
        hh, mm = (int(x) for x in report_at.split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError
    except Exception:
        raise ValueError(f"reporting.weekly_report_at_utc must be 'HH:MM' (24h UTC), got '{report_at}'")
    report_day = str(rep_raw.get("weekly_report_day", "mon")).lower()
    _WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    if report_day not in _WEEKDAYS:
        raise ValueError(f"reporting.weekly_report_day must be one of {_WEEKDAYS}, got '{report_day}'")
    reporting = ReportingConfig(
        equity_sample_interval_minutes=int(rep_raw.get("equity_sample_interval_minutes", 60)),
        weekly_report_day=report_day,
        weekly_report_at_utc=report_at,
        report_accounts=[str(x) for x in (rep_raw.get("report_accounts") or [])],
    )

    pm_raw = raw.get("position_monitor", {}) or {}
    position_monitor = PositionMonitorConfig(
        enabled=bool(pm_raw.get("enabled", False)),
        profit_threshold_pct=float(pm_raw.get("profit_threshold_pct", 25.0)),
        check_interval_minutes=int(pm_raw.get("check_interval_minutes", 60)),
        alert_throttle_hours=int(pm_raw.get("alert_throttle_hours", 24)),
    )

    return AppConfig(
        signals=signals,
        exchanges=exchanges,
        execution=execution,
        server=server,
        signal_generator=signal_generator,
        notifications=notifications,
        reporting=reporting,
        position_monitor=position_monitor,
    )
