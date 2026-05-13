"""Configuration management for Cognifold."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """LLM model configuration."""

    name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_exploration_steps: int = 3


@dataclass
class ScoringWeights:
    """Weights for context window scoring."""

    alpha: float = 0.4  # PageRank weight
    beta: float = 0.4  # Recency weight
    gamma: float = 0.2  # Access frequency weight
    decay_rate: float = 0.01  # Recency decay rate (per hour)


@dataclass
class ContextConfig:
    """Context window configuration."""

    max_nodes: int = 20  # Default max nodes in context
    min_score_threshold: float = 0.01


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None


@dataclass
class FastModeConfig:
    """Configuration for --fast layered pipeline."""

    enabled: bool = False
    batch_size: int = 10
    skip_embeddings: bool = False


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation (concept merging)."""

    enabled: bool = False  # OFF by default (safe)
    similarity_threshold: float = 0.85  # cosine similarity for merge candidates
    min_graph_size: int = 30  # skip consolidation if fewer nodes
    max_merges_per_pass: float = 0.1  # max fraction of concept count per pass
    merge_strategy: str = "highest_access"  # "highest_access" or "most_recent"
    auto_trigger_every_n_events: int = 50  # 0 = manual only

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}"
            )
        if not 0.0 < self.max_merges_per_pass <= 1.0:
            raise ValueError(
                f"max_merges_per_pass must be in (0, 1], got {self.max_merges_per_pass}"
            )
        if self.merge_strategy not in ("highest_access", "most_recent"):
            raise ValueError(
                f"merge_strategy must be 'highest_access' or 'most_recent', got '{self.merge_strategy}'"
            )
        if self.min_graph_size < 0:
            raise ValueError(f"min_graph_size must be >= 0, got {self.min_graph_size}")
        if self.auto_trigger_every_n_events < 0:
            raise ValueError(
                f"auto_trigger_every_n_events must be >= 0, got {self.auto_trigger_every_n_events}"
            )


@dataclass
class LifecycleConfig:
    """Configuration for node lifecycle management (archival and forgetting)."""

    enabled: bool = False  # OFF by default
    archive_after_hours: float = 168.0  # 7 days idle
    prune_after_hours: float = 720.0  # 30 days in archived state
    min_access_for_keep: int = 3  # accessed 3+ times = never auto-archive
    resolved_intent_archive_hours: float = 48.0  # resolved intents archive after 2 days

    def __post_init__(self) -> None:
        if self.archive_after_hours <= 0:
            raise ValueError(f"archive_after_hours must be > 0, got {self.archive_after_hours}")
        if self.prune_after_hours <= 0:
            raise ValueError(f"prune_after_hours must be > 0, got {self.prune_after_hours}")
        if self.prune_after_hours <= self.archive_after_hours:
            raise ValueError(
                f"prune_after_hours ({self.prune_after_hours}) must exceed "
                f"archive_after_hours ({self.archive_after_hours})"
            )
        if self.min_access_for_keep < 0:
            raise ValueError(f"min_access_for_keep must be >= 0, got {self.min_access_for_keep}")
        if self.resolved_intent_archive_hours <= 0:
            raise ValueError(
                f"resolved_intent_archive_hours must be > 0, got {self.resolved_intent_archive_hours}"
            )


@dataclass
class TraceConfig:
    """Configuration for cognitive trace collection."""

    enabled: bool = False  # OFF by default
    max_entries: int = 1000  # ring buffer size
    persist_mode: str = "memory"  # "memory" or "jsonl"

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {self.max_entries}")
        if self.persist_mode not in ("memory", "jsonl"):
            raise ValueError(f"persist_mode must be 'memory' or 'jsonl', got '{self.persist_mode}'")


@dataclass
class CognifoldConfig:
    """Main configuration for Cognifold.

    Configuration is loaded from (in order of precedence):
    1. Environment variables (COGNIFOLD_*)
    2. YAML config file
    3. Default values
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"

    # API key (always from environment)
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Load API key from environment or thread-local scope."""
        from cognifold.service.llm_keys import get_api_key

        self.api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("OPENAI_API_KEY") or ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> CognifoldConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            CognifoldConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> CognifoldConfig:
        """Load configuration from environment variables.

        Environment variables are prefixed with COGNIFOLD_.
        Nested config uses double underscore: COGNIFOLD_MODEL__TEMPERATURE

        Returns:
            CognifoldConfig instance.
        """
        data: dict[str, Any] = {}

        # Model config
        if temp := os.environ.get("COGNIFOLD_MODEL__TEMPERATURE"):
            data.setdefault("model", {})["temperature"] = float(temp)
        if name := os.environ.get("COGNIFOLD_MODEL__NAME"):
            data.setdefault("model", {})["name"] = name
        if max_tokens := os.environ.get("COGNIFOLD_MODEL__MAX_TOKENS"):
            data.setdefault("model", {})["max_tokens"] = int(max_tokens)
        if max_steps := os.environ.get("COGNIFOLD_MODEL__MAX_EXPLORATION_STEPS"):
            data.setdefault("model", {})["max_exploration_steps"] = int(max_steps)

        # Scoring config
        if alpha := os.environ.get("COGNIFOLD_SCORING__ALPHA"):
            data.setdefault("scoring", {})["alpha"] = float(alpha)
        if beta := os.environ.get("COGNIFOLD_SCORING__BETA"):
            data.setdefault("scoring", {})["beta"] = float(beta)
        if gamma := os.environ.get("COGNIFOLD_SCORING__GAMMA"):
            data.setdefault("scoring", {})["gamma"] = float(gamma)

        # Context config
        if max_nodes := os.environ.get("COGNIFOLD_CONTEXT__MAX_NODES"):
            data.setdefault("context", {})["max_nodes"] = int(max_nodes)

        # Logging config
        if level := os.environ.get("COGNIFOLD_LOGGING__LEVEL"):
            data.setdefault("logging", {})["level"] = level
        if log_file := os.environ.get("COGNIFOLD_LOGGING__FILE"):
            data.setdefault("logging", {})["file"] = log_file

        # Consolidation config
        if cons_enabled := os.environ.get("COGNIFOLD_CONSOLIDATION__ENABLED"):
            data.setdefault("consolidation", {})["enabled"] = cons_enabled.lower() in (
                "true",
                "1",
                "yes",
            )
        if cons_threshold := os.environ.get("COGNIFOLD_CONSOLIDATION__SIMILARITY_THRESHOLD"):
            data.setdefault("consolidation", {})["similarity_threshold"] = float(cons_threshold)

        # Lifecycle config
        if lc_enabled := os.environ.get("COGNIFOLD_LIFECYCLE__ENABLED"):
            data.setdefault("lifecycle", {})["enabled"] = lc_enabled.lower() in ("true", "1", "yes")
        if lc_archive := os.environ.get("COGNIFOLD_LIFECYCLE__ARCHIVE_AFTER_HOURS"):
            data.setdefault("lifecycle", {})["archive_after_hours"] = float(lc_archive)

        # Trace config
        if tr_enabled := os.environ.get("COGNIFOLD_TRACE__ENABLED"):
            data.setdefault("trace", {})["enabled"] = tr_enabled.lower() in ("true", "1", "yes")
        if tr_max := os.environ.get("COGNIFOLD_TRACE__MAX_ENTRIES"):
            data.setdefault("trace", {})["max_entries"] = int(tr_max)

        # Paths
        if data_dir := os.environ.get("COGNIFOLD_DATA_DIR"):
            data["data_dir"] = data_dir
        if output_dir := os.environ.get("COGNIFOLD_OUTPUT_DIR"):
            data["output_dir"] = output_dir

        return cls._from_dict(data)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> CognifoldConfig:
        """Load configuration from file and/or environment.

        Environment variables override file values.

        Args:
            config_path: Optional path to YAML config file.

        Returns:
            CognifoldConfig instance.
        """
        # Start with defaults
        config = cls()

        # Load from YAML if provided
        if config_path:
            path = Path(config_path)
            if path.exists():
                config = cls.from_yaml(path)

        # Override with environment variables
        env_config = cls.from_env()
        config = cls._merge(config, env_config)

        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CognifoldConfig:
        """Create config from dictionary."""
        model_data = data.get("model", {})
        scoring_data = data.get("scoring", {})
        context_data = data.get("context", {})
        logging_data = data.get("logging", {})
        consolidation_data = data.get("consolidation", {})
        lifecycle_data = data.get("lifecycle", {})
        trace_data = data.get("trace", {})

        return cls(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            scoring=ScoringWeights(**scoring_data) if scoring_data else ScoringWeights(),
            context=ContextConfig(**context_data) if context_data else ContextConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
            consolidation=ConsolidationConfig(**consolidation_data)
            if consolidation_data
            else ConsolidationConfig(),
            lifecycle=LifecycleConfig(**lifecycle_data) if lifecycle_data else LifecycleConfig(),
            trace=TraceConfig(**trace_data) if trace_data else TraceConfig(),
            data_dir=data.get("data_dir", "data"),
            output_dir=data.get("output_dir", "output"),
        )

    @classmethod
    def _merge(cls, base: CognifoldConfig, override: CognifoldConfig) -> CognifoldConfig:
        """Merge two configs, with override taking precedence for non-default values."""
        # For simplicity, just use override's values if they differ from defaults
        default = cls()

        return cls(
            model=ModelConfig(
                name=override.model.name
                if override.model.name != default.model.name
                else base.model.name,
                temperature=override.model.temperature
                if override.model.temperature != default.model.temperature
                else base.model.temperature,
                max_tokens=override.model.max_tokens
                if override.model.max_tokens != default.model.max_tokens
                else base.model.max_tokens,
                max_exploration_steps=override.model.max_exploration_steps
                if override.model.max_exploration_steps != default.model.max_exploration_steps
                else base.model.max_exploration_steps,
            ),
            scoring=ScoringWeights(
                alpha=override.scoring.alpha
                if override.scoring.alpha != default.scoring.alpha
                else base.scoring.alpha,
                beta=override.scoring.beta
                if override.scoring.beta != default.scoring.beta
                else base.scoring.beta,
                gamma=override.scoring.gamma
                if override.scoring.gamma != default.scoring.gamma
                else base.scoring.gamma,
                decay_rate=override.scoring.decay_rate
                if override.scoring.decay_rate != default.scoring.decay_rate
                else base.scoring.decay_rate,
            ),
            context=ContextConfig(
                max_nodes=override.context.max_nodes
                if override.context.max_nodes != default.context.max_nodes
                else base.context.max_nodes,
                min_score_threshold=override.context.min_score_threshold
                if override.context.min_score_threshold != default.context.min_score_threshold
                else base.context.min_score_threshold,
            ),
            logging=LoggingConfig(
                level=override.logging.level
                if override.logging.level != default.logging.level
                else base.logging.level,
                format=override.logging.format
                if override.logging.format != default.logging.format
                else base.logging.format,
                file=override.logging.file
                if override.logging.file != default.logging.file
                else base.logging.file,
            ),
            consolidation=ConsolidationConfig(
                enabled=override.consolidation.enabled
                if override.consolidation.enabled != default.consolidation.enabled
                else base.consolidation.enabled,
                similarity_threshold=override.consolidation.similarity_threshold
                if override.consolidation.similarity_threshold
                != default.consolidation.similarity_threshold
                else base.consolidation.similarity_threshold,
                min_graph_size=override.consolidation.min_graph_size
                if override.consolidation.min_graph_size != default.consolidation.min_graph_size
                else base.consolidation.min_graph_size,
                max_merges_per_pass=override.consolidation.max_merges_per_pass
                if override.consolidation.max_merges_per_pass
                != default.consolidation.max_merges_per_pass
                else base.consolidation.max_merges_per_pass,
                merge_strategy=override.consolidation.merge_strategy
                if override.consolidation.merge_strategy != default.consolidation.merge_strategy
                else base.consolidation.merge_strategy,
                auto_trigger_every_n_events=override.consolidation.auto_trigger_every_n_events
                if override.consolidation.auto_trigger_every_n_events
                != default.consolidation.auto_trigger_every_n_events
                else base.consolidation.auto_trigger_every_n_events,
            ),
            lifecycle=LifecycleConfig(
                enabled=override.lifecycle.enabled
                if override.lifecycle.enabled != default.lifecycle.enabled
                else base.lifecycle.enabled,
                archive_after_hours=override.lifecycle.archive_after_hours
                if override.lifecycle.archive_after_hours != default.lifecycle.archive_after_hours
                else base.lifecycle.archive_after_hours,
                prune_after_hours=override.lifecycle.prune_after_hours
                if override.lifecycle.prune_after_hours != default.lifecycle.prune_after_hours
                else base.lifecycle.prune_after_hours,
                min_access_for_keep=override.lifecycle.min_access_for_keep
                if override.lifecycle.min_access_for_keep != default.lifecycle.min_access_for_keep
                else base.lifecycle.min_access_for_keep,
                resolved_intent_archive_hours=override.lifecycle.resolved_intent_archive_hours
                if override.lifecycle.resolved_intent_archive_hours
                != default.lifecycle.resolved_intent_archive_hours
                else base.lifecycle.resolved_intent_archive_hours,
            ),
            trace=TraceConfig(
                enabled=override.trace.enabled
                if override.trace.enabled != default.trace.enabled
                else base.trace.enabled,
                max_entries=override.trace.max_entries
                if override.trace.max_entries != default.trace.max_entries
                else base.trace.max_entries,
                persist_mode=override.trace.persist_mode
                if override.trace.persist_mode != default.trace.persist_mode
                else base.trace.persist_mode,
            ),
            data_dir=override.data_dir if override.data_dir != default.data_dir else base.data_dir,
            output_dir=override.output_dir
            if override.output_dir != default.output_dir
            else base.output_dir,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the configuration.
        """
        data = {
            "model": {
                "name": self.model.name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "max_exploration_steps": self.model.max_exploration_steps,
            },
            "scoring": {
                "alpha": self.scoring.alpha,
                "beta": self.scoring.beta,
                "gamma": self.scoring.gamma,
                "decay_rate": self.scoring.decay_rate,
            },
            "context": {
                "max_nodes": self.context.max_nodes,
                "min_score_threshold": self.context.min_score_threshold,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
            },
            "consolidation": {
                "enabled": self.consolidation.enabled,
                "similarity_threshold": self.consolidation.similarity_threshold,
                "min_graph_size": self.consolidation.min_graph_size,
                "max_merges_per_pass": self.consolidation.max_merges_per_pass,
                "merge_strategy": self.consolidation.merge_strategy,
                "auto_trigger_every_n_events": self.consolidation.auto_trigger_every_n_events,
            },
            "lifecycle": {
                "enabled": self.lifecycle.enabled,
                "archive_after_hours": self.lifecycle.archive_after_hours,
                "prune_after_hours": self.lifecycle.prune_after_hours,
                "min_access_for_keep": self.lifecycle.min_access_for_keep,
                "resolved_intent_archive_hours": self.lifecycle.resolved_intent_archive_hours,
            },
            "trace": {
                "enabled": self.trace.enabled,
                "max_entries": self.trace.max_entries,
                "persist_mode": self.trace.persist_mode,
            },
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
