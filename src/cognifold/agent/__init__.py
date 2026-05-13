"""LangGraph agent for Cognifold."""

from cognifold.agent.agent import CognifoldAgent
from cognifold.agent.config import AgentConfig
from cognifold.agent.context import AgentContext, ContextNode
from cognifold.agent.domain import (
    COMPUTER_ACTIVITY_DOMAIN,
    DOMAIN_REGISTRY,
    PERSONAL_TIMELINE_DOMAIN,
    SERVICE_LOGS_DOMAIN,
    WIKI_DOMAIN,
    DomainConfig,
    get_domain_config,
    register_domain,
)
from cognifold.agent.prompt_profile import PromptProfile, load_prompt_profiles
from cognifold.agent.prompts import (
    ReasoningMode,
    format_system_prompt,
    format_system_prompt_for_domain,
    format_user_prompt,
)
from cognifold.agent.tools import GraphTools

__all__ = [
    "COMPUTER_ACTIVITY_DOMAIN",
    "DOMAIN_REGISTRY",
    "PERSONAL_TIMELINE_DOMAIN",
    "SERVICE_LOGS_DOMAIN",
    "WIKI_DOMAIN",
    "AgentConfig",
    "AgentContext",
    "CognifoldAgent",
    "ContextNode",
    "DomainConfig",
    "GraphTools",
    "PromptProfile",
    "ReasoningMode",
    "format_system_prompt",
    "format_system_prompt_for_domain",
    "format_user_prompt",
    "get_domain_config",
    "load_prompt_profiles",
    "register_domain",
]
