"""
logging_config.py
------------------
Structured logging for api.py.

Two sinks:
  1. stdout, one JSON object per line -- easy to pipe into any log
     aggregator (CloudWatch, Datadog, `docker logs | jq`, etc.) without a
     separate log-shipping config.
  2. logs/api_requests.jsonl -- same JSON records, kept on disk, matching
     the existing repo convention (rag_latency_log.jsonl, logs/*.csv) of
     always leaving a local audit trail from a plain `python`/`uvicorn` run.

Every request gets a short request_id (first 8 chars of a uuid4) so a
single request's "request started" / "request finished" / any error log
lines can be grepped out together.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "api_requests.jsonl")


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # anything passed via logger.info(..., extra={...}) gets merged in
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "levelname", "levelno", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno",
                       "funcName", "created", "msecs", "relativeCreated", "thread",
                       "threadName", "processName", "process", "name", "taskName"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("rag_api")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # avoid duplicate handlers on uvicorn --reload re-imports
        return logger

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]
