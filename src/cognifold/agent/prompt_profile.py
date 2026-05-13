from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cognifold.agent.config import AgentConfig
from cognifold.agent.prompts import ReasoningMode


@dataclass(frozen=True)
class PromptProfile:
    profile_id: str
    domain: str | None = None
    mode: ReasoningMode | None = None

    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_exploration_steps: int | None = None

    concept_guidelines: tuple[str, ...] | None = None
    action_guidelines: tuple[str, ...] | None = None
    time_guidelines: tuple[str, ...] | None = None

    system_prompt_template: str | None = None
    user_prompt_template: str | None = None

    # Section composition (Phase 13)
    disabled_sections: frozenset[str] | None = None
    extra_sections: dict[str, str] | None = None

    features: dict[str, Any] = field(default_factory=dict)

    def to_agent_config(self, base: AgentConfig | None = None) -> AgentConfig:
        if base is None:
            base = AgentConfig()

        return AgentConfig(
            model_name=self.model_name or base.model_name,
            temperature=self.temperature if self.temperature is not None else base.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else base.max_tokens,
            max_exploration_steps=self.max_exploration_steps
            if self.max_exploration_steps is not None
            else base.max_exploration_steps,
            concept_guidelines=self.concept_guidelines or base.concept_guidelines,
            action_guidelines=self.action_guidelines or base.action_guidelines,
            time_guidelines=self.time_guidelines or base.time_guidelines,
        )


def load_prompt_profiles(path: str | Path) -> dict[str, PromptProfile]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    profiles_data = data.get("profiles") or {}
    profiles: dict[str, PromptProfile] = {}

    for profile_id, cfg in profiles_data.items():
        cfg = cfg or {}

        mode_value = cfg.get("mode")
        mode = ReasoningMode(mode_value) if mode_value else None

        model = cfg.get("model") or {}
        guidelines = cfg.get("guidelines") or {}
        templates = cfg.get("templates") or {}
        sections_cfg = cfg.get("sections") or {}

        disabled_raw = sections_cfg.get("disabled")
        disabled = frozenset(disabled_raw) if disabled_raw else None
        extra_raw = sections_cfg.get("extra")
        extra = dict(extra_raw) if extra_raw else None

        profiles[profile_id] = PromptProfile(
            profile_id=profile_id,
            domain=cfg.get("domain"),
            mode=mode,
            model_name=model.get("name"),
            temperature=model.get("temperature"),
            max_tokens=model.get("max_tokens"),
            max_exploration_steps=model.get("max_exploration_steps"),
            concept_guidelines=tuple(guidelines.get("concept") or ()) or None,
            action_guidelines=tuple(guidelines.get("action") or ()) or None,
            time_guidelines=tuple(guidelines.get("time") or ()) or None,
            system_prompt_template=templates.get("system"),
            user_prompt_template=templates.get("user"),
            disabled_sections=disabled,
            extra_sections=extra,
            features=cfg.get("features") or {},
        )

    return profiles
