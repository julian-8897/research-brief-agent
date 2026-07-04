from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.settings import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s"
        if settings.structured_logs
        else "%(levelname)s:%(name)s:%(message)s",
    )


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, default=str, sort_keys=True))
