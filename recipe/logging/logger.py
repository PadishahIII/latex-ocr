from pathlib import Path
from typing import Any, Dict, Optional

from recipe.logging import LoggerConfig, LoggerFactory, LogFormat
import logging
import logging.handlers

from recipe.logging.factory import ConfigurableJSONFormatter


_logger: Optional[logging.Logger] = None


def default_logger_config() -> LoggerConfig:
    return LoggerConfig(
        name="default_logger",
        log_dir="logs",
        max_bytes=10 * 1024 * 1024,
        backup_count=0,
        log_level=logging.INFO,
        file_format=LogFormat.JSON,
        console_format=LogFormat.RAW,
        enable_console=True,
        enable_file=True,
        include_source=True,
        include_traceback=False,
    )


def get_logger_by_config(config: LoggerConfig) -> logging.Logger:
    if _logger:
        return _logger
    return LoggerFactory.get_logger_from_config(config)


def get_logger_by_name(name: str) -> logging.Logger:
    logger_config = default_logger_config().model_copy()
    logger_config.name = name
    logger = get_logger_by_config(logger_config)
    return logger


def get_logger(package: str | None, module_name: str) -> logging.Logger:
    """Get a logger with name '<package>.<module>'"""
    if package:
        name = f"{package}.{module_name}"
    else:
        name = f"UnknownPackage.{module_name}"
    return get_logger_by_name(name)


def get_ctx_logger(config: LoggerConfig, ctx: Dict[str, Any]) -> logging.Logger:
    return ContextLogger(config, ctx)


class ContextLogger(logging.Logger):
    """
    A logger class that supports injecting context into all log operations
    and configurable traceback and source file info logging.

    This class inherits from logging.Logger and adds context injection via extra fields,
    configurable traceback inclusion (for WARNING and above levels), and configurable source file info.
    """

    def __init__(self, config: LoggerConfig, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the ContextLogger.

        Args:
            config: LoggerConfig instance with all settings
            context: Dictionary of context to inject into all log messages
        """
        super().__init__(config.name, config.log_level)

        self.config = config
        self.context = context or {}

        # Clear any existing handlers
        self.handlers[:] = []

        # Set up handlers based on config
        if config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(config.log_level)
            if config.console_format == LogFormat.JSON:
                console_formatter = ConfigurableJSONFormatter(
                    config.include_source, config.include_traceback
                )
            else:
                console_formatter = logging.Formatter(
                    fmt="{asctime} [{levelname}] {name}: {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            console_handler.setFormatter(console_formatter)
            self.addHandler(console_handler)

        if config.enable_file:
            # Ensure log directory exists
            log_dir = config.log_dir or "logs"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # Create log file path
            log_file = Path(log_dir) / "server.log"

            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(config.log_level)
            if config.file_format == LogFormat.JSON:
                file_formatter = ConfigurableJSONFormatter(
                    config.include_source, config.include_traceback
                )
            else:
                file_formatter = logging.Formatter(
                    fmt="{asctime} [{levelname}] {name}: {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            file_handler.setFormatter(file_formatter)
            self.addHandler(file_handler)

        # Prevent propagation to root logger
        self.propagate = False

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        """
        Override _log to inject context and handle configuration.

        Adds context to extra fields and conditionally includes stack_info
        for WARNING and above levels when include_traceback is enabled.
        """
        # Inject context into extra
        if extra is None:
            extra = {}
        elif not isinstance(extra, dict):
            # Convert Mapping to dict if necessary
            extra = dict(extra)

        # Update with context
        extra.update(self.context)

        # Add stack_info if traceback is enabled AND log level is WARNING or higher
        if self.config.include_traceback:
            # Only include stack_info for WARNING and more critical levels
            # WARNING=30, ERROR=40, CRITICAL=50
            if level >= logging.WARNING:
                stack_info = True

        # Call parent _log with modified parameters
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
