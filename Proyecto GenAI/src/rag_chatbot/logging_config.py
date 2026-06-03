import logging
from logging.handlers import RotatingFileHandler

from rag_chatbot.config import AppSettings, get_settings


LOGGER_NAME = "rag_chatbot"
LOG_FILE_NAME = "rag_chatbot.log"


def configure_logging(settings: AppSettings | None = None) -> logging.Logger:
    settings = settings or get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, settings.log_level, logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.log_dir / LOG_FILE_NAME,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug("Logging configured")
    return logger
