"""
Logging factory module for centralized logger configuration.

This module provides a factory for creating loggers with rotating file handlers
and configurable settings. It supports:
- Named loggers with specified configurations
- Rotating file handlers with user-specified directory and rotation size
- Multiple formatters (verbose, simple, json)
- Console and file output options
"""

import logging
import logging.handlers
import json
from pathlib import Path
from typing import Optional, Dict, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class LogFormat(str, Enum):
    """Supported log formats."""

    JSON = "json"
    RAW = "raw"


class LoggerConfig(BaseModel):
    """
    Pydantic model for logger configuration.

    This model provides type safety and validation for logger settings,
    ensuring all configuration parameters are valid before creating loggers.
    """

    name: str = Field(..., description="Logger name (typically module name)")
    log_dir: Optional[str] = Field(
        None,
        description="Directory for log files. If None, uses 'logs' in current directory",
    )
    max_bytes: int = Field(
        10 * 1024 * 1024,
        gt=0,
        description="Maximum size of each log file before rotation (default: 10MB)",
    )
    backup_count: int = Field(
        5, ge=0, description="Number of backup files to keep (default: 5)"
    )
    log_level: Union[int, str] = Field(
        logging.INFO, description="Logging level as int (logging.INFO) or str ('INFO')"
    )
    file_format: LogFormat = Field(
        LogFormat.JSON, description="Format for file output - JSON or RAW"
    )
    console_format: LogFormat = Field(
        LogFormat.RAW, description="Format for console output - JSON or RAW"
    )
    enable_console: bool = Field(True, description="Whether to enable console logging")
    enable_file: bool = Field(True, description="Whether to enable file logging")
    include_source: bool = Field(
        True, description="Whether to include source file information in logs"
    )
    include_traceback: bool = Field(
        False,
        description="Whether to include traceback information in logs for WARNING and above levels",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate and normalize log level."""
        if isinstance(v, str):
            level_name = v.upper()
            if not hasattr(logging, level_name):
                raise ValueError(f"Invalid log level: {v}")
            return getattr(logging, level_name)
        elif isinstance(v, int):
            # Validate that it's a valid logging level
            valid_levels = [
                logging.DEBUG,
                logging.INFO,
                logging.WARNING,
                logging.ERROR,
                logging.CRITICAL,
            ]
            if v not in valid_levels:
                raise ValueError(
                    f"Invalid log level: {v}. Must be one of {valid_levels}"
                )
            return v
        else:
            raise ValueError(f"Log level must be string or int, got {type(v)}")

    @field_validator("log_dir")
    @classmethod
    def validate_log_dir(cls, v):
        """Validate log directory path."""
        if v is not None:
            # Convert to Path and validate it's not a file
            path = Path(v)
            if path.exists() and path.is_file():
                raise ValueError(f"Log directory path points to a file: {v}")
        return v

    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter that includes detailed attributes.

    This formatter outputs log records as JSON with comprehensive metadata
    including source file, line number, timestamp, and other useful attributes.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        # Create the base log data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "pathname": record.pathname,
            "filename": record.filename,
            "process": record.process,
            "process_name": getattr(record, "processName", None),
            "thread": record.thread,
            "thread_name": getattr(record, "threadName", None),
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add stack information if present and for WARNING level or higher
        if (
            hasattr(record, "stack_info")
            and record.stack_info
            and record.levelno >= logging.WARNING
        ):
            log_data["stack_info"] = self.formatStack(record.stack_info)

        # Add any extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False, separators=(",", ":"))


class ConfigurableJSONFormatter(logging.Formatter):
    """
    Configurable JSON formatter that includes detailed attributes based on settings.

    This formatter outputs log records as JSON with configurable source file info and traceback.
    """

    def __init__(
        self,
        include_source: bool = True,
        include_traceback: bool = False,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ):
        super().__init__(datefmt=datefmt)
        self.include_source = include_source
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        # Create the base log data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_source:
            log_data.update(
                {
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "pathname": record.pathname,
                    "filename": record.filename,
                    "process": record.process,
                    "process_name": getattr(record, "processName", None),
                    "thread": record.thread,
                    "thread_name": getattr(record, "threadName", None),
                }
            )

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add stack information if present and enabled for WARNING and above levels
        if (
            self.include_traceback
            and hasattr(record, "stack_info")
            and record.stack_info
            and record.levelno >= logging.WARNING
        ):
            log_data["stack_info"] = self.formatStack(record.stack_info)

        # Add any extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False, separators=(",", ":"))


class LoggerFactory:
    """
    Factory class for creating and managing loggers with rotating file handlers.

    This factory provides centralized logger configuration with support for:
    - Rotating file handlers with configurable size and backup count
    - Multiple output formats (verbose, simple, json)
    - Console and file output options
    - Per-logger configuration caching
    """

    _loggers: Dict[str, logging.Logger] = {}
    _formatters: Dict[str, logging.Formatter] = {}

    @classmethod
    def _get_formatter(cls, format_type: Union[LogFormat, str]) -> logging.Formatter:
        """
        Get or create a formatter of the specified type.

        Args:
            format_type: The type of formatter to create (LogFormat enum or string)

        Returns:
            Configured logging formatter
        """
        # Handle both LogFormat enum and string values
        if isinstance(format_type, LogFormat):
            format_value = format_type.value
            format_enum = format_type
        else:
            format_value = format_type
            format_enum = LogFormat(format_type)

        if format_value not in cls._formatters:
            if format_enum == LogFormat.JSON:
                formatter = JSONFormatter(datefmt="%Y-%m-%d %H:%M:%S")
            elif format_enum == LogFormat.RAW:
                formatter = logging.Formatter(
                    fmt="{asctime} [{levelname}] {name}: {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            else:
                raise ValueError(f"Unsupported format type: {format_type}")

            cls._formatters[format_value] = formatter

        return cls._formatters[format_value]

    @classmethod
    def _create_config(cls, **kwargs) -> LoggerConfig:
        """
        Create a LoggerConfig from keyword arguments.

        Args:
            **kwargs: Logger configuration parameters

        Returns:
            Validated LoggerConfig instance
        """
        return LoggerConfig(**kwargs)

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,
        log_level: Union[int, str] = logging.INFO,
        file_format: LogFormat = LogFormat.JSON,
        console_format: LogFormat = LogFormat.RAW,
        enable_console: bool = True,
        enable_file: bool = True,
        include_source: bool = True,
        include_traceback: bool = False,
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.

        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files. If None, uses 'logs' in current directory
            max_bytes: Maximum size of each log file before rotation (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
            log_level: Logging level as int (logging.INFO) or str ("INFO") (default: logging.INFO)
            file_format: Format for file output - JSON or RAW (default: JSON)
            console_format: Format for console output - JSON or RAW (default: RAW)
            enable_console: Whether to enable console logging (default: True)
            enable_file: Whether to enable file logging (default: True)

        Returns:
            Configured logger instance
        """
        # Create and validate configuration using Pydantic model
        config = cls._create_config(
            name=name,
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            log_level=log_level,
            file_format=file_format,
            console_format=console_format,
            enable_console=enable_console,
            enable_file=enable_file,
            include_source=include_source,
            include_traceback=include_traceback,
        )

        return cls.get_logger_from_config(config)

    @classmethod
    def get_logger_from_config(cls, config: LoggerConfig) -> logging.Logger:
        """
        Get or create a logger using a LoggerConfig instance.

        Args:
            config: Validated LoggerConfig instance

        Returns:
            Configured logger instance
        """
        # Create cache key based on configuration
        cache_key = f"{config.name}_{config.log_dir}_{config.max_bytes}_{config.backup_count}_{config.log_level}_{config.file_format}_{config.console_format}_{config.enable_console}_{config.enable_file}_{config.include_source}_{config.include_traceback}"

        if cache_key in cls._loggers:
            return cls._loggers[cache_key]

        # Create new logger
        logger = logging.getLogger(config.name)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Set log level
        logger.setLevel(config.log_level)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # Add console handler if enabled
        if config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(config.log_level)
            if config.console_format == LogFormat.JSON:
                console_formatter = ConfigurableJSONFormatter(
                    config.include_source, config.include_traceback
                )
            elif config.console_format == LogFormat.RAW:
                console_formatter = logging.Formatter(
                    fmt="{asctime} [{levelname}] {name}: {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            else:
                raise ValueError(
                    f"Unsupported console format type: {config.console_format}"
                )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Add file handler if enabled
        if config.enable_file:
            # Ensure log directory exists
            log_dir = config.log_dir or "logs"

            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Create log file path
            log_file = log_path / "server.log"

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
            elif config.file_format == LogFormat.RAW:
                file_formatter = logging.Formatter(
                    fmt="{asctime} [{levelname}] {name}: {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            else:
                raise ValueError(f"Unsupported file format type: {config.file_format}")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Cache the logger
        cls._loggers[cache_key] = logger

        return logger

    @classmethod
    def create_logger_config(
        cls,
        name: str,
        log_dir: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        log_level: Union[int, str] = logging.INFO,
        file_format: LogFormat = LogFormat.JSON,
        console_format: LogFormat = LogFormat.RAW,
        enable_console: bool = True,
        enable_file: bool = True,
        include_source: bool = True,
        include_traceback: bool = False,
    ) -> LoggerConfig:
        """
        Create a LoggerConfig instance for logger settings.

        This can be used to store and reuse logger configurations.

        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files
            max_bytes: Maximum size of each log file before rotation
            backup_count: Number of backup files to keep
            log_level: Logging level
            file_format: Format for file output
            console_format: Format for console output
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging

        Returns:
            Validated LoggerConfig instance
        """
        return LoggerConfig(
            name=name,
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            log_level=log_level,
            file_format=file_format,
            console_format=console_format,
            enable_console=enable_console,
            enable_file=enable_file,
            include_source=include_source,
            include_traceback=include_traceback,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the logger cache.

        This will force recreation of loggers on next access.
        Useful for testing or when configuration changes.
        """
        cls._loggers.clear()
        cls._formatters.clear()


# Convenience functions for common use cases
def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    log_level: Union[int, str] = logging.INFO,
    file_format: LogFormat = LogFormat.JSON,
    console_format: LogFormat = LogFormat.RAW,
    include_source: bool = True,
    include_traceback: bool = False,
) -> logging.Logger:
    """
    Convenience function to get a logger with default settings.

    Args:
        name: Logger name
        log_dir: Directory for log files (optional)
        max_bytes: Maximum size of each log file before rotation
        log_level: Logging level as int (logging.INFO) or str ("INFO")
        file_format: Format for file output - JSON or RAW (default: JSON)
        console_format: Format for console output - JSON or RAW (default: RAW)

    Returns:
        Configured logger instance
    """
    return LoggerFactory.get_logger(
        name=name,
        log_dir=log_dir,
        max_bytes=max_bytes,
        log_level=log_level,
        file_format=file_format,
        console_format=console_format,
        include_source=include_source,
        include_traceback=include_traceback,
    )


def get_module_logger(
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    log_level: Union[int, str] = logging.INFO,
    file_format: LogFormat = LogFormat.JSON,
    console_format: LogFormat = LogFormat.RAW,
    include_source: bool = True,
    include_traceback: bool = False,
) -> logging.Logger:
    """
    Convenience function to get a logger for the calling module.

    Args:
        log_dir: Directory for log files (optional)
        max_bytes: Maximum size of each log file before rotation
        log_level: Logging level as int (logging.INFO) or str ("INFO")
        file_format: Format for file output - JSON or RAW (default: JSON)
        console_format: Format for console output - JSON or RAW (default: RAW)

    Returns:
        Configured logger instance with the calling module's name
    """
    import inspect

    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        module_name = caller_frame.f_globals.get("__name__", "unknown")
        return LoggerFactory.get_logger(
            name=module_name,
            log_dir=log_dir,
            max_bytes=max_bytes,
            log_level=log_level,
            file_format=file_format,
            console_format=console_format,
            include_source=include_source,
            include_traceback=include_traceback,
        )
    finally:
        del frame
