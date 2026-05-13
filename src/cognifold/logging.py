"""Logging configuration for Cognifold.

Uses structlog for structured JSON logging when available, falls back to stdlib
logging. Existing code using ``logging.getLogger(__name__)`` automatically emits
structured JSON via structlog's stdlib integration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.config import LoggingConfig

# Module-level logger
_logger: logging.Logger | None = None
_structured: bool = False


def _try_setup_structlog(level: int, json_output: bool = True) -> bool:
    """Configure structlog to wrap stdlib logging.

    Returns True if structlog was configured, False if not available.
    """
    try:
        import structlog  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root handler to use structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure the root logger so ALL stdlib loggers emit structured output
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    return True


def setup_logging(
    config: LoggingConfig | None = None,
    *,
    json_output: bool | None = None,
) -> logging.Logger:
    """Set up logging for Cognifold.

    When structlog is installed, all stdlib loggers automatically emit
    structured JSON. Falls back to plain-text formatting otherwise.

    Args:
        config: Logging configuration. Uses defaults if not provided.
        json_output: Force JSON (True) or console (False) output.
            Default: True when structlog is available.

    Returns:
        The root Cognifold logger.
    """
    global _logger, _structured

    if _logger is not None:
        return _logger

    # Resolve level
    level_name = config.level if config else "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)

    # Try structlog first
    use_json = json_output if json_output is not None else True
    _structured = _try_setup_structlog(level, json_output=use_json)

    # Create the cognifold logger
    logger = logging.getLogger("cognifold")
    logger.setLevel(level)

    if not _structured:
        # Fallback: plain-text stdlib logging
        fmt = config.format if config else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (always plain-text, independent of structlog)
    if config and config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_fmt))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid double output)
    logger.propagate = _structured

    _logger = logger
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for a Cognifold component.

    Args:
        name: Component name. If None, returns the root logger.

    Returns:
        Logger instance.
    """
    if _logger is None:
        setup_logging()

    if name:
        return logging.getLogger(f"cognifold.{name}")
    return logging.getLogger("cognifold")


def bind_contextvars(**kwargs: Any) -> None:
    """Bind key-value pairs to the structlog context for the current task/request.

    No-op if structlog is not available.
    """
    try:
        import structlog  # pyright: ignore[reportMissingImports]

        structlog.contextvars.bind_contextvars(**kwargs)
    except ImportError:
        pass


def clear_contextvars() -> None:
    """Clear all structlog context variables for the current task/request.

    No-op if structlog is not available.
    """
    try:
        import structlog  # pyright: ignore[reportMissingImports]

        structlog.contextvars.clear_contextvars()
    except ImportError:
        pass


class LogContext:
    """Context manager for structured logging of operations."""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs: str | int | float):
        """Initialize log context.

        Args:
            logger: Logger to use.
            operation: Name of the operation being logged.
            **kwargs: Additional context to include in logs.
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs

    def __enter__(self) -> LogContext:
        """Enter the context and log start."""
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.info(f"Starting {self.operation} {context_str}".strip())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context and log completion or error."""
        if exc_val:
            self.logger.error(f"Failed {self.operation}: {exc_val}")
        else:
            self.logger.info(f"Completed {self.operation}")


def log_event_processing(event_id: str, event_title: str) -> LogContext:
    """Create a log context for event processing.

    Args:
        event_id: ID of the event being processed.
        event_title: Title of the event.

    Returns:
        LogContext for the operation.
    """
    logger = get_logger("simulator")
    return LogContext(logger, "event processing", event_id=event_id, title=event_title)


def log_plan_execution(plan_id: str, operation_count: int) -> LogContext:
    """Create a log context for plan execution.

    Args:
        plan_id: ID of the plan being executed.
        operation_count: Number of operations in the plan.

    Returns:
        LogContext for the operation.
    """
    logger = get_logger("executor")
    return LogContext(logger, "plan execution", plan_id=plan_id, operations=operation_count)


def log_agent_call(event_id: str) -> LogContext:
    """Create a log context for agent LLM call.

    Args:
        event_id: ID of the event being analyzed.

    Returns:
        LogContext for the operation.
    """
    logger = get_logger("agent")
    return LogContext(logger, "agent analysis", event_id=event_id)
