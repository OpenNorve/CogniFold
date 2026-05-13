"""Domain registration and listing endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/domains", tags=["domains"])


class DomainRegisterRequest(BaseModel):
    """Request body for registering a new domain."""

    name: str
    description: str
    event_description: str
    node_type_descriptions: dict[str, str] = {}
    concept_examples: list[dict[str, Any]] = []
    action_examples: list[dict[str, Any]] = []
    time_examples: list[dict[str, Any]] = []
    pattern_types: list[str] = []
    hierarchy_examples: list[dict[str, str]] = []
    concept_guidelines: list[str] = []
    action_guidelines: list[str] = []
    time_guidelines: list[str] = []


@router.post("", status_code=status.HTTP_201_CREATED)
async def register_domain(body: DomainRegisterRequest) -> dict[str, str]:
    """Register a new domain configuration."""
    from cognifold.agent.domain import DomainConfig, register_domain

    config = DomainConfig(
        name=body.name,
        description=body.description,
        event_description=body.event_description,
        node_type_descriptions=body.node_type_descriptions,
        concept_examples=body.concept_examples,
        action_examples=body.action_examples,
        time_examples=body.time_examples,
        pattern_types=body.pattern_types,
        hierarchy_examples=body.hierarchy_examples,
        concept_guidelines=tuple(body.concept_guidelines),
        action_guidelines=tuple(body.action_guidelines),
        time_guidelines=tuple(body.time_guidelines),
    )
    register_domain(config)
    return {"status": "registered", "name": body.name}


@router.get("")
async def list_domains() -> dict[str, list[str]]:
    """List all registered domain names."""
    from cognifold.agent.domain import DOMAIN_REGISTRY

    return {"domains": list(DOMAIN_REGISTRY.keys())}


@router.get("/{name}")
async def get_domain(name: str) -> dict[str, str]:
    """Get details of a specific domain."""
    from cognifold.agent.domain import get_domain_config

    try:
        config = get_domain_config(name)
        return {"name": config.name, "description": config.description}
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{name}' not found",
        ) from None
