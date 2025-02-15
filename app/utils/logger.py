import logging
import logging.config
import sys
import json
import os

def configure_logging():
    """
    Configure logging for Fluent Bit (log to file)
    """
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": json.dumps({"level": "%(levelname)s", "message": "%(message)s"})
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/app.log",
                "when": "midnight",
                "formatter": "json"
            }
        },
        "loggers": {
            "": {
                "handlers": ["file"],
                "level": "INFO"
            }
        }
    })