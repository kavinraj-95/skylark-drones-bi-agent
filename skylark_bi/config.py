"""Runtime configuration, loaded from the environment.

Everything the app needs to reach monday.com and an LLM lives here. Nothing in this
module knows anything about the *business* data - board IDs and credentials only.

On Streamlit Community Cloud these come from the app's Secrets; locally from a .env
file (see .env.example). We read `st.secrets` opportunistically so that the core
package stays importable - and unit-testable - without Streamlit installed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache"

#: monday API version to pin. Versions rotate quarterly (yyyy-mm); pinning keeps us
#: insulated from breaking changes rolled out on the floating "current" version.
DEFAULT_MONDAY_API_VERSION = "2026-07"
MONDAY_API_URL = "https://api.monday.com/v2"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader - avoids a dependency for one trivial job.

    Only sets keys that are not already in the environment, so a real env var always
    wins over the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _secret(name: str, default: str = "") -> str:
    """Read a setting from env, then Streamlit secrets, then a default.

    Streamlit is only consulted when it is *already* imported - i.e. we are running
    inside the app. Importing it here would emit a bare-mode warning on every CLI
    invocation and pull a heavy dependency into unit tests for nothing.
    """
    value = os.environ.get(name)
    if value:
        return value.strip()

    streamlit = sys.modules.get("streamlit")
    if streamlit is not None:  # pragma: no cover - only inside a Streamlit runtime
        try:
            raw = streamlit.secrets.get(name)
            if raw:
                return str(raw).strip()
        except Exception:
            # No secrets.toml, or no such key. Both are fine.
            pass
    return default


def _int_secret(name: str, default: int) -> int:
    try:
        return int(_secret(name, str(default)))
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised when required configuration is missing.

    Carries a message aimed at whoever is deploying the app, not at an end user.
    """


@dataclass(frozen=True)
class MondayConfig:
    api_token: str
    api_version: str
    deals_board_id: str = ""
    work_orders_board_id: str = ""
    deals_board_name: str = "Deals"
    work_orders_board_name: str = "Work Orders"

    @property
    def needs_board_discovery(self) -> bool:
        """True when either board ID is absent and must be looked up by name."""
        return not (self.deals_board_id and self.work_orders_board_id)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    monday: MondayConfig
    llm: LLMConfig
    data_ttl_seconds: int = 900
    fiscal_year_start_month: int = 4
    cache_dir: Path = field(default=CACHE_DIR)


def load_settings(*, require_llm: bool = True) -> Settings:
    """Build `Settings` from the environment.

    Raises `ConfigError` with actionable text when something required is absent -
    a missing token should surface as a clear banner in the UI, never a stack trace.
    """
    _load_dotenv(REPO_ROOT / ".env")

    token = _secret("MONDAY_API_TOKEN")
    if not token:
        raise ConfigError(
            "MONDAY_API_TOKEN is not set. Create a personal API token in monday.com "
            "(avatar → Developers → My access tokens) and add it to your .env file "
            "locally, or to the app's Secrets when deploying."
        )

    monday = MondayConfig(
        api_token=token,
        api_version=_secret("MONDAY_API_VERSION", DEFAULT_MONDAY_API_VERSION),
        deals_board_id=_secret("MONDAY_DEALS_BOARD_ID"),
        work_orders_board_id=_secret("MONDAY_WORK_ORDERS_BOARD_ID"),
        deals_board_name=_secret("MONDAY_DEALS_BOARD_NAME", "Deals"),
        work_orders_board_name=_secret("MONDAY_WORK_ORDERS_BOARD_NAME", "Work Orders"),
    )

    provider = _secret("LLM_PROVIDER", "gemini").lower()
    if provider == "anthropic":
        llm_key = _secret("ANTHROPIC_API_KEY")
        llm_model = _secret("ANTHROPIC_MODEL", "claude-sonnet-5")
        key_hint = (
            "ANTHROPIC_API_KEY is not set. This must be a Claude Platform key from "
            "console.anthropic.com - a Claude Pro/Max subscription cannot authenticate "
            "a hosted server app."
        )
    else:
        provider = "gemini"
        llm_key = _secret("GEMINI_API_KEY")
        llm_model = _secret("GEMINI_MODEL", "gemini-flash-lite-latest")
        key_hint = (
            "GEMINI_API_KEY is not set. Create one free at https://aistudio.google.com/apikey"
        )

    if require_llm and not llm_key:
        raise ConfigError(key_hint)

    return Settings(
        monday=monday,
        llm=LLMConfig(provider=provider, api_key=llm_key, model=llm_model),
        data_ttl_seconds=_int_secret("DATA_TTL_SECONDS", 900),
        fiscal_year_start_month=_int_secret("FISCAL_YEAR_START_MONTH", 4),
    )
