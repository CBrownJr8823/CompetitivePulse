from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIRECTORY: Final[Path] = PROJECT_ROOT / "reports"
DEFAULT_LOGS_DIRECTORY: Final[Path] = PROJECT_ROOT / "logs"


def _environment_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _environment_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, received {raw_value!r}."
        ) from exc

    if value < minimum:
        raise ValueError(
            f"Environment variable {name} must be at least {minimum}, received {value}."
        )

    return value


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    log_level: str
    request_timeout_seconds: int
    max_retries: int
    max_competitors: int
    user_agent: str
    firecrawl_api_key: str | None
    langsmith_tracing_enabled: bool
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str
    reports_directory: Path
    logs_directory: Path

    @property
    def firecrawl_enabled(self) -> bool:
        return bool(self.firecrawl_api_key)

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing_enabled and bool(self.langsmith_api_key)


def load_settings() -> Settings:
    load_dotenv(override=False)

    log_level = os.getenv("COMPETITIVE_PULSE_LOG_LEVEL", "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        raise ValueError(
            "COMPETITIVE_PULSE_LOG_LEVEL must be one of "
            f"{sorted(valid_levels)}, received {log_level!r}."
        )

    return Settings(
        app_name=os.getenv("COMPETITIVE_PULSE_APP_NAME", "CompetitivePulse"),
        environment=os.getenv("COMPETITIVE_PULSE_ENVIRONMENT", "development"),
        log_level=log_level,
        request_timeout_seconds=_environment_int(
            "COMPETITIVE_PULSE_REQUEST_TIMEOUT_SECONDS",
            default=20,
            minimum=3,
        ),
        max_retries=_environment_int(
            "COMPETITIVE_PULSE_MAX_RETRIES",
            default=3,
            minimum=1,
        ),
        max_competitors=_environment_int(
            "COMPETITIVE_PULSE_MAX_COMPETITORS",
            default=10,
            minimum=1,
        ),
        user_agent=os.getenv(
            "COMPETITIVE_PULSE_USER_AGENT",
            "CompetitivePulse/1.0 (+https://github.com/your-org/competitivepulse)",
        ),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY") or None,
        langsmith_tracing_enabled=_environment_bool("LANGCHAIN_TRACING_V2"),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY") or None,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "competitivepulse"),
        langsmith_endpoint=os.getenv(
            "LANGCHAIN_ENDPOINT",
            "https://api.smith.langchain.com",
        ),
        reports_directory=DEFAULT_REPORTS_DIRECTORY,
        logs_directory=DEFAULT_LOGS_DIRECTORY,
    )


def configure_observability(settings: Settings) -> None:
    """
    Configures LangSmith solely through standard environment variables.

    LangGraph and LangChain automatically detect these values and emit nested
    traces covering graph nodes, retries, timing, and compatible model calls.
    """
    if not settings.langsmith_enabled:
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logging.getLogger(__name__).info(
        "LangSmith tracing enabled for project '%s'.",
        settings.langsmith_project,
    )


def configure_logging(settings: Settings) -> logging.Logger:
    settings.logs_directory.mkdir(parents=True, exist_ok=True)

    log_file = settings.logs_directory / "competitivepulse.log"
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    logger = logging.getLogger(settings.app_name)
    logger.info(
        "Logging initialized | environment=%s | log_file=%s",
        settings.environment,
        log_file,
    )
    return logger
