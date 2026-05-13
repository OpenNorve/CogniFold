"""Service logs event stream generator."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cognifold.generator.base import BaseEventGenerator


@dataclass
class ServiceTopology:
    """Defines a service architecture for event generation.

    Attributes:
        name: Topology name (e.g., "ecommerce", "saas_platform").
        description: Brief description of the system.
        services: List of service names in the system.
        external_dependencies: External services/APIs used.
        traffic_patterns: Typical traffic patterns.
        common_operations: Common business operations.
    """

    name: str
    description: str
    services: list[str] = field(default_factory=list)
    external_dependencies: list[str] = field(default_factory=list)
    traffic_patterns: list[str] = field(default_factory=list)
    common_operations: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert topology to prompt format."""
        return f"""## Service Topology: {self.name}
{self.description}

Services:
{chr(10).join(f"- {s}" for s in self.services)}

External Dependencies:
{chr(10).join(f"- {d}" for d in self.external_dependencies)}

Traffic Patterns:
{chr(10).join(f"- {p}" for p in self.traffic_patterns)}

Common Operations:
{chr(10).join(f"- {o}" for o in self.common_operations)}"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "services": self.services,
            "external_dependencies": self.external_dependencies,
            "traffic_patterns": self.traffic_patterns,
            "common_operations": self.common_operations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceTopology:
        """Create from dictionary."""
        return cls(**data)


# Sample service topologies
SAMPLE_TOPOLOGIES: dict[str, ServiceTopology] = {
    "ecommerce": ServiceTopology(
        name="ecommerce",
        description="E-commerce platform with product catalog, orders, and payments",
        services=[
            "api-gateway",
            "product-service",
            "inventory-service",
            "order-service",
            "payment-service",
            "user-service",
            "notification-service",
            "search-service",
        ],
        external_dependencies=[
            "Stripe (payments)",
            "Elasticsearch (search)",
            "Redis (caching)",
            "PostgreSQL (database)",
            "SendGrid (email)",
            "Twilio (SMS)",
        ],
        traffic_patterns=[
            "Morning spike (9-10am) as users check orders",
            "Lunch rush (12-1pm) with increased browsing",
            "Evening peak (7-9pm) highest transaction volume",
            "Periodic inventory sync every 15 minutes",
            "Nightly batch jobs for analytics",
        ],
        common_operations=[
            "Product search and browse",
            "Add to cart and checkout",
            "Payment processing",
            "Order status updates",
            "Inventory adjustments",
            "User registration and login",
        ],
    ),
    "saas_platform": ServiceTopology(
        name="saas_platform",
        description="B2B SaaS platform with multi-tenant architecture",
        services=[
            "api-gateway",
            "auth-service",
            "tenant-service",
            "billing-service",
            "analytics-service",
            "workflow-service",
            "notification-service",
            "file-service",
        ],
        external_dependencies=[
            "Auth0 (authentication)",
            "Stripe (billing)",
            "AWS S3 (file storage)",
            "PostgreSQL (database)",
            "Redis (caching/queues)",
            "Datadog (monitoring)",
        ],
        traffic_patterns=[
            "Business hours peak (9am-5pm weekdays)",
            "API rate limiting spikes",
            "Batch processing at midnight UTC",
            "Webhook deliveries throughout day",
            "Monthly billing cycle end-of-month",
        ],
        common_operations=[
            "User authentication and authorization",
            "Tenant provisioning",
            "API calls from integrations",
            "File uploads and downloads",
            "Workflow executions",
            "Usage metering and billing",
        ],
    ),
    "microservices_demo": ServiceTopology(
        name="microservices_demo",
        description="Simple microservices demo with user and post services",
        services=[
            "api-gateway",
            "user-service",
            "post-service",
            "comment-service",
            "notification-service",
        ],
        external_dependencies=[
            "PostgreSQL (database)",
            "Redis (caching)",
            "RabbitMQ (messaging)",
        ],
        traffic_patterns=[
            "Steady traffic during day",
            "Occasional traffic spikes",
            "Background job processing",
        ],
        common_operations=[
            "User CRUD operations",
            "Post creation and retrieval",
            "Comment threads",
            "Push notifications",
        ],
    ),
}


def get_service_topology(name: str) -> ServiceTopology:
    """Get a service topology by name.

    Args:
        name: Topology name.

    Returns:
        The service topology.

    Raises:
        KeyError: If topology not found.
    """
    if name not in SAMPLE_TOPOLOGIES:
        available = ", ".join(SAMPLE_TOPOLOGIES.keys())
        raise KeyError(f"Unknown topology: {name}. Available: {available}")
    return SAMPLE_TOPOLOGIES[name]


class ServiceLogsGenerator(BaseEventGenerator):
    """Generates service log events using Gemini LLM.

    Creates realistic sequences of microservice log events including HTTP
    requests, database operations, message queue events, and system events.

    Example:
        >>> from cognifold.generator.service_logs import (
        ...     ServiceLogsGenerator,
        ...     get_service_topology,
        ... )
        >>> topology = get_service_topology("ecommerce")
        >>> generator = ServiceLogsGenerator()
        >>> events = generator.generate(topology=topology, num_events=100)
    """

    source_name = "service-logs"

    def generate(
        self,
        num_events: int = 100,
        start_date: datetime | None = None,
        num_days: int = 3,
        topology: ServiceTopology | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a timeline of service log events.

        Args:
            num_events: Target number of events to generate.
            start_date: Starting date for events (defaults to today).
            num_days: Number of days to span events across.
            topology: The service topology to generate events for (required).
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries in timeline format.

        Raises:
            ValueError: If topology is not provided.
        """
        if topology is None:
            raise ValueError("topology is required for ServiceLogsGenerator")

        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        all_events: list[dict[str, Any]] = []
        events_per_day = num_events // num_days

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            day_events = self._generate_day(
                date=current_date,
                target_events=events_per_day,
                previous_events=all_events[-10:] if all_events else [],
                topology=topology,
            )
            all_events.extend(day_events)

        # Re-number event IDs and add source
        for i, event in enumerate(all_events):
            event["event_id"] = f"svc-{i + 1:03d}"
            event["source"] = self.source_name

        return all_events[:num_events]

    def _generate_day(
        self,
        date: datetime,
        target_events: int,
        previous_events: list[dict[str, Any]],
        topology: ServiceTopology | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate service log events for a single day.

        Args:
            date: The date to generate events for.
            target_events: Target number of events for this day.
            previous_events: Recent events for context continuity.
            topology: The service topology to generate events for.
            max_retries: Maximum number of retry attempts on failure.
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries for the day.
        """
        import time

        if topology is None:
            return []

        client = self._ensure_client()
        from google.genai import types

        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")

        prev_context = self._build_prev_context(previous_events)

        prompt = self._build_generation_prompt(
            date_str=date_str,
            day_name=day_name,
            target_events=target_events,
            prev_context=prev_context,
            topology=topology,
        )

        gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=gen_config,
                )

                if not response.candidates:
                    raise ValueError(f"No candidates in response for {date_str}")

                candidate = response.candidates[0]

                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)
                    if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                        raise ValueError(f"Response blocked: {finish_reason}")

                if not candidate.content or not candidate.content.parts:
                    raise ValueError(f"No content parts in response for {date_str}")

                text = candidate.content.parts[0].text
                events = self._parse_events(text, date)

                if not events:
                    preview = text[:500] if len(text) > 500 else text
                    raise ValueError(f"Failed to parse events for {date_str}. Preview: {preview!r}")

                return events

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    import sys

                    print(f"Error generating events for {date_str}: {e}", file=sys.stderr)
                    return []

        return []

    def _build_generation_prompt(
        self,
        date_str: str,
        day_name: str,
        target_events: int,
        prev_context: str,
        topology: ServiceTopology | None = None,
        **kwargs: Any,
    ) -> str:
        """Build the prompt for service log event generation.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            day_name: Day of week name.
            target_events: Target number of events.
            prev_context: Context from previous events.
            topology: The service topology to generate events for.
            **kwargs: Additional arguments (unused).

        Returns:
            The prompt string for the LLM.
        """
        if topology is None:
            raise ValueError("topology is required for prompt generation")

        return f"""You are a service log event stream generator. Generate realistic microservice log events.

{topology.to_prompt()}

## Task
Generate exactly {target_events} service log events for {day_name}, {date_str}.

## Event Types (use dot notation)
- **http.request**: Incoming HTTP request to a service
- **http.response**: HTTP response sent
- **http.error**: HTTP error (4xx, 5xx)
- **db.query**: Database query execution
- **db.connection**: Database connection event
- **cache.hit**: Cache hit
- **cache.miss**: Cache miss
- **cache.set**: Cache entry set
- **queue.publish**: Message published to queue
- **queue.consume**: Message consumed from queue
- **queue.error**: Queue processing error
- **auth.login**: User authentication
- **auth.token_refresh**: Token refresh
- **auth.logout**: User logout
- **business.order_created**: Order created (e-commerce)
- **business.payment_processed**: Payment processed
- **business.user_signup**: User registration
- **system.startup**: Service startup
- **system.shutdown**: Service shutdown
- **system.health_check**: Health check
- **system.config_reload**: Configuration reload
- **ops.deployment**: Deployment event
- **ops.scale**: Auto-scaling event
- **ops.alert**: Alert triggered

## Requirements
1. Events must show realistic request flows across services
2. Include correlated events (same trace_id for related events)
3. Show realistic latencies and status codes
4. Include some errors and retries
5. Timestamps should be throughout the day with realistic patterns
6. Show inter-service communication

## Output Format
Return a JSON array. Each event must have:
- event_id: unique ID (format: "svc-XXX")
- timestamp: ISO 8601 datetime
- event_type: one of the types above
- title: short description
- description: log message or context
- context: structured log data

## Context Field Examples

http.request:
{{"service": "api-gateway", "method": "POST", "endpoint": "/api/orders", "trace_id": "abc123", "user_id": "u-456"}}

http.response:
{{"service": "api-gateway", "status": 201, "latency_ms": 45, "trace_id": "abc123"}}

db.query:
{{"service": "order-service", "query_type": "INSERT", "table": "orders", "duration_ms": 12, "trace_id": "abc123"}}

queue.publish:
{{"service": "order-service", "queue": "order-events", "event_type": "order.created", "trace_id": "abc123"}}

business.order_created:
{{"service": "order-service", "order_id": "ord-789", "total": 99.99, "items": 3, "trace_id": "abc123"}}

ops.alert:
{{"service": "payment-service", "alert_name": "HighErrorRate", "severity": "warning", "threshold": "5%", "current": "7.2%"}}

## Example Events

{{
  "event_id": "svc-001",
  "timestamp": "{date_str}T09:15:32.123Z",
  "event_type": "http.request",
  "title": "POST /api/orders",
  "description": "New order request received",
  "context": {{"service": "api-gateway", "method": "POST", "endpoint": "/api/orders", "trace_id": "tr-001", "user_id": "u-123"}}
}}

{{
  "event_id": "svc-002",
  "timestamp": "{date_str}T09:15:32.145Z",
  "event_type": "db.query",
  "title": "Check inventory",
  "description": "SELECT from inventory table",
  "context": {{"service": "inventory-service", "query_type": "SELECT", "table": "inventory", "duration_ms": 8, "trace_id": "tr-001"}}
}}

{{
  "event_id": "svc-003",
  "timestamp": "{date_str}T09:15:32.200Z",
  "event_type": "business.order_created",
  "title": "Order created",
  "description": "Successfully created order ord-456",
  "context": {{"service": "order-service", "order_id": "ord-456", "total": 149.99, "items": 2, "trace_id": "tr-001"}}
}}
{prev_context}

Generate {target_events} events as a JSON array:"""

    def _parse_events(self, text: str, date: datetime) -> list[dict[str, Any]]:
        """Parse LLM response into event dictionaries.

        Args:
            text: Raw LLM response text.
            date: The date for these events.

        Returns:
            List of parsed event dictionaries.
        """
        # Remove markdown code blocks if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Extract JSON array from response
        json_match = re.search(r"\[[\s\S]*\]", text)
        if not json_match:
            return []

        json_str = json_match.group(0)

        # Fix common JSON issues
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}", "}", json_str)

        try:
            events = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        # Validate and normalize events
        validated_events = []
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue

            # Ensure required fields
            if "event_id" not in event:
                event["event_id"] = f"svc-{uuid.uuid4().hex[:8]}"

            if "timestamp" not in event:
                # Generate timestamp distributed through day
                hour = i * 24 // len(events)
                minute = (i * 60) % 60
                event["timestamp"] = date.replace(hour=hour, minute=minute).isoformat() + "Z"

            if "event_type" not in event:
                event["event_type"] = "http.request"

            if "title" not in event:
                event["title"] = "Service event"

            if "context" not in event:
                event["context"] = {}

            validated_events.append(event)

        return validated_events

    def save_timeline(
        self,
        events: list[dict[str, Any]],
        path: str | Path,
        timeline_id: str | None = None,
        description: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        *,
        topology: ServiceTopology | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save generated events to a timeline JSON file.

        Args:
            events: List of event dictionaries.
            path: Path to save the timeline.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata (from base class).
            topology: Optional service topology to include in metadata.

        Returns:
            Path to the saved file.
        """
        merged_metadata = dict(extra_metadata) if extra_metadata else {}
        if topology:
            merged_metadata["topology"] = topology.to_dict()

        return super().save_timeline(
            events=events,
            path=path,
            timeline_id=timeline_id,
            description=description,
            extra_metadata=merged_metadata,
        )
