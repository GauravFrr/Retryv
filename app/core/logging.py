import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging format and log level for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger("retryv")
