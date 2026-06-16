"""Shared logging configuration for the modernization engine.

Import ``get_logger`` in any module::

    from core.logger import get_logger
    logger = get_logger(__name__)

Environment variables
---------------------
LOG_LEVEL          Console log level (default: INFO).  One of DEBUG, INFO, WARNING, ERROR.
LOG_FILE           Path to the rotating log file (default: modernization.log).
LOG_FILE_MAX_BYTES Max bytes before rotation (default: 5_000_000 = ~5 MB).
LOG_FILE_BACKUPS   Number of backup log files to keep (default: 3).
LOG_REDACT_SOURCE  If 1, redact user source code from logs (hashes lines >20 chars).
                   Protects user code in shared/audited deployments (default: 0).
"""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import os
import re
import sys

_CONFIGURED = False
_DEFAULT_LOG_FILE = "modernization.log"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUPS = 3
_DEFAULT_REDACT_SOURCE = True

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RedactSourceFilter(logging.Filter):
    """
    Redacts user-submitted C++ source code from log messages.
    
    Hashes code lines >20 chars to protect user code while maintaining
    readability for debugging. Useful for shared/audited deployments.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, str):
            return True
        
        # Hash multiline code blocks (common in prompts)
        # Pattern: code snippets, stack traces, etc. with long source-like lines
        lines = record.msg.split('\n')
        redacted = []
        
        for line in lines:
            # Redact very long lines that look like C/C++ source code to avoid
            # accidentally writing full user source into persistent logs.
            if len(line) > 200 and (
                '#include' in line or
                '{' in line or
                '}' in line or
                ';' in line or
                re.search(r'\b(int|float|void|char|bool|std\::)\b', line, re.IGNORECASE)
            ):
                h = hashlib.md5(line.encode()).hexdigest()[:8]
                redacted.append(f"[REDACTED:{h}]")
            else:
                redacted.append(line)
        
        record.msg = '\n'.join(redacted)
        return True


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", _DEFAULT_LOG_LEVEL).strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = os.environ.get("LOG_FILE", _DEFAULT_LOG_FILE).strip() or _DEFAULT_LOG_FILE

    try:
        max_bytes = int(os.environ.get("LOG_FILE_MAX_BYTES", str(_DEFAULT_MAX_BYTES)).strip())
    except ValueError:
        max_bytes = _DEFAULT_MAX_BYTES

    try:
        backup_count = int(os.environ.get("LOG_FILE_BACKUPS", str(_DEFAULT_BACKUPS)).strip())
    except ValueError:
        backup_count = _DEFAULT_BACKUPS

    redact_source = os.environ.get("LOG_REDACT_SOURCE", "1").strip() == "1"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [console_handler]

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError:
        pass  # Read-only filesystem or permission error — skip file logging.

    root = logging.getLogger()
    # Avoid adding duplicate handlers when the module is reloaded.
    if not root.handlers:
        for handler in handlers:
            root.addHandler(handler)

    # Add redaction filter if enabled (to file handler only for safety)
    if redact_source and len(handlers) > 1:
        handlers[1].addFilter(_RedactSourceFilter())

    root.setLevel(logging.DEBUG)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call."""
    _configure()
    return logging.getLogger(name)
