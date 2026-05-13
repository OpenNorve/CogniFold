"""Domain configuration for multi-domain support.

This module provides domain-specific configurations that customize the agent's
prompts and behavior for different event stream domains (personal timeline,
computer activity, service logs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainConfig:
    """Configuration for a specific domain/use case.

    Attributes:
        name: Domain identifier (e.g., "personal-timeline", "computer-activity").
        description: Brief description of what this domain represents.
        event_description: How events are described in this domain.
        node_type_descriptions: Descriptions for each node type in this domain.
            Supports both "intent" and legacy "action" keys.
        concept_examples: Example concepts for this domain.
        action_examples: Example intents for this domain (legacy name, maps to intents).
        time_examples: Example time nodes for this domain.
        pattern_types: Types of patterns to look for in this domain.
        hierarchy_examples: Examples of concept hierarchies.
        concept_guidelines: Domain-specific concept discovery guidelines.
        action_guidelines: Domain-specific intent discovery guidelines (legacy name).
        time_guidelines: Domain-specific time node guidelines.
        disabled_sections: Section or group names to exclude from the system
            prompt (e.g., ``frozenset({"intents"})`` disables all intent
            sections).  Default: empty (all sections enabled).
        extra_sections: Custom sections to inject into the prompt.  Keys are
            section names, values are prompt text.
        extra_section_position: Where to inject *extra_sections*:
            ``"before_rules"`` (default), ``"after_tools"``, ``"after_rules"``.

    Note:
        "action" terminology is preserved for backward compatibility but
        semantically these now represent "intents" (goals/desires) that can
        be converted to concrete actions by the Intent-to-Action agent.
    """

    name: str
    description: str
    event_description: str
    node_type_descriptions: dict[str, str] = field(default_factory=dict)
    concept_examples: list[dict[str, Any]] = field(default_factory=list)
    action_examples: list[dict[str, Any]] = field(default_factory=list)
    time_examples: list[dict[str, Any]] = field(default_factory=list)
    pattern_types: list[str] = field(default_factory=list)
    hierarchy_examples: list[dict[str, str]] = field(default_factory=list)
    concept_guidelines: tuple[str, ...] = field(default_factory=tuple)
    action_guidelines: tuple[str, ...] = field(default_factory=tuple)
    time_guidelines: tuple[str, ...] = field(default_factory=tuple)
    # Section composition (Phase 13)
    disabled_sections: frozenset[str] = field(default_factory=frozenset)
    extra_sections: dict[str, str] = field(default_factory=dict)
    extra_section_position: str = "before_rules"
    opt_in_sections: frozenset[str] = field(default_factory=frozenset)


# Personal Timeline Domain
PERSONAL_TIMELINE_DOMAIN = DomainConfig(
    name="personal-timeline",
    description="Personal life events and daily activities",
    event_description="daily activities (meals, work, exercise, social, rest, etc.)",
    node_type_descriptions={
        "event": "Direct representations of activities (meals, work, exercise, etc.)",
        "concept": "Higher-level patterns that emerge from activities (e.g., 'Morning Routine', 'Fitness Habit', 'Work Focus')",
        "intent": "Goals, desires, or intentions (e.g., 'Research sleep improvement', 'Prepare for meeting')",
        "action": "Goals, desires, or intentions (legacy alias for 'intent')",
        "time": "Temporal anchors representing deadlines, scheduled events, or recurring periods",
    },
    concept_examples=[
        {
            "concept_id": "c-morning-routine",
            "title": "Morning Routine",
            "description": "Consistent morning pattern with coffee and exercise",
            "strength": 0.7,
            "evidence_count": 5,
        },
        {
            "concept_id": "c-fitness-habit",
            "title": "Fitness Habit",
            "description": "Regular exercise pattern, gym visits 3x/week",
            "strength": 0.8,
            "evidence_count": 8,
        },
    ],
    action_examples=[
        {
            "action_id": "a-morning-coffee",
            "title": "Make morning coffee",
            "description": "Part of established morning routine",
            "priority": "medium",
            "pattern_source": "c-morning-routine",
        },
        {
            "action_id": "a-gym-session",
            "title": "Go to gym",
            "description": "Regular workout based on fitness habit",
            "priority": "medium",
            "pattern_source": "c-fitness-habit",
        },
    ],
    time_examples=[
        {
            "id": "t-meeting-2pm",
            "title": "Team Meeting at 2pm",
            "scheduled_time": "2026-01-18T14:00:00Z",
            "recurrence": "weekly",
        },
        {
            "id": "t-project-deadline",
            "title": "Project Deadline",
            "scheduled_time": "2026-01-20T17:00:00Z",
            "urgency_window_hours": 48,
        },
    ],
    pattern_types=[
        "Daily routines (morning, evening)",
        "Weekly patterns (gym days, meeting days)",
        "Social patterns (who you meet, where)",
        "Activity clusters (work focus, relaxation)",
    ],
    hierarchy_examples=[
        {"level1": "Morning Coffee", "level2": "Caffeine Ritual", "level3": "Self-Care Routine"},
        {"level1": "Gym Workout", "level2": "Fitness Habit", "level3": "Health Management"},
    ],
    concept_guidelines=(
        "Look for repeated activities at similar times (routines)",
        "Identify emotional or energy patterns (focus periods, rest needs)",
        "Notice social interaction patterns",
        "Track location-based habits (home, office, gym)",
        "Recognize meal patterns and dietary habits",
    ),
    action_guidelines=(
        "Suggest preparation actions before scheduled events",
        "Predict next occurrence of habitual activities",
        "Create reminders based on past patterns",
        "Suggest break times during long focus periods",
    ),
    time_guidelines=(
        "Create TIME nodes for upcoming meetings and appointments",
        "Track project deadlines mentioned in work events",
        "Recognize recurring time patterns (daily standups, weekly reviews)",
    ),
)


# Computer Activity Domain
COMPUTER_ACTIVITY_DOMAIN = DomainConfig(
    name="computer-activity",
    description="Computer usage events and productivity patterns",
    event_description="computer activities (app usage, browsing, coding, communication, etc.)",
    node_type_descriptions={
        "event": "Direct representations of computer actions (app launches, page visits, file operations)",
        "concept": "Higher-level patterns in computer usage (e.g., 'Development Focus', 'Research Session', 'Communication Block')",
        "action": "Productivity suggestions (e.g., 'Take a break', 'Review pending PRs', 'Schedule focus time')",
        "time": "Temporal anchors for work schedules, deadlines, and productivity windows",
    },
    concept_examples=[
        {
            "concept_id": "c-dev-focus",
            "title": "Development Focus",
            "description": "Deep coding session with IDE and terminal",
            "strength": 0.8,
            "evidence_count": 12,
        },
        {
            "concept_id": "c-research-pattern",
            "title": "Research Session",
            "description": "Multi-tab browsing with documentation and Stack Overflow",
            "strength": 0.6,
            "evidence_count": 5,
        },
        {
            "concept_id": "c-communication-block",
            "title": "Communication Block",
            "description": "Slack/email checking pattern after meetings",
            "strength": 0.7,
            "evidence_count": 8,
        },
    ],
    action_examples=[
        {
            "action_id": "a-take-break",
            "title": "Take a break",
            "description": "Extended focus time detected, suggest break",
            "priority": "medium",
            "pattern_source": "c-dev-focus",
        },
        {
            "action_id": "a-review-prs",
            "title": "Review pending PRs",
            "description": "Morning productivity window - good time for code review",
            "priority": "medium",
            "suggested_time": "2026-01-19T09:00:00Z",
        },
    ],
    time_examples=[
        {
            "id": "t-focus-window",
            "title": "Morning Focus Window",
            "scheduled_time": "2026-01-18T09:00:00Z",
            "recurrence": "daily",
        },
        {
            "id": "t-pr-deadline",
            "title": "PR Review Deadline",
            "scheduled_time": "2026-01-18T17:00:00Z",
        },
    ],
    pattern_types=[
        "Focus sessions (IDE-heavy, no distractions)",
        "Research patterns (documentation, Stack Overflow)",
        "Communication patterns (Slack, email bursts)",
        "Context switching (between projects/tasks)",
        "Productivity windows (high/low activity periods)",
    ],
    hierarchy_examples=[
        {"level1": "VS Code Editing", "level2": "Development Work", "level3": "Productivity Flow"},
        {
            "level1": "Slack Messages",
            "level2": "Team Communication",
            "level3": "Collaboration Pattern",
        },
    ],
    concept_guidelines=(
        "Identify focus sessions (sustained IDE/editor usage)",
        "Detect context switching patterns (too many app switches)",
        "Recognize research vs implementation phases",
        "Track communication patterns and interruptions",
        "Notice productivity rhythms across the day",
    ),
    action_guidelines=(
        "Suggest breaks after extended focus periods",
        "Recommend batch processing for communications",
        "Identify optimal times for deep work",
        "Alert on excessive context switching",
    ),
    time_guidelines=(
        "Track scheduled meetings from calendar apps",
        "Identify recurring productivity windows",
        "Recognize deadline-related activity spikes",
    ),
)


# Service Logs Domain
SERVICE_LOGS_DOMAIN = DomainConfig(
    name="service-logs",
    description="Microservice and infrastructure events",
    event_description="service events (HTTP requests, database operations, queue messages, deployments)",
    node_type_descriptions={
        "event": "Direct representations of service operations (requests, queries, messages)",
        "concept": "Higher-level patterns in service behavior (e.g., 'High Traffic Period', 'Error Cascade', 'Latency Spike')",
        "action": "Operational suggestions (e.g., 'Scale up service', 'Investigate error', 'Clear cache')",
        "time": "Temporal anchors for maintenance windows, deployment schedules, SLA deadlines",
    },
    concept_examples=[
        {
            "concept_id": "c-traffic-spike",
            "title": "High Traffic Period",
            "description": "Consistent traffic increase during lunch hours",
            "strength": 0.9,
            "evidence_count": 50,
        },
        {
            "concept_id": "c-error-cascade",
            "title": "Error Cascade Pattern",
            "description": "Payment service errors triggering order failures",
            "strength": 0.7,
            "evidence_count": 8,
        },
        {
            "concept_id": "c-db-latency",
            "title": "Database Latency Spike",
            "description": "Query slowdowns during batch processing",
            "strength": 0.6,
            "evidence_count": 12,
        },
    ],
    action_examples=[
        {
            "action_id": "a-scale-up",
            "title": "Scale up payment-service",
            "description": "High error rate detected, consider scaling",
            "priority": "high",
            "pattern_source": "c-error-cascade",
        },
        {
            "action_id": "a-investigate-latency",
            "title": "Investigate DB latency",
            "description": "Recurring slow queries during batch jobs",
            "priority": "medium",
            "pattern_source": "c-db-latency",
        },
    ],
    time_examples=[
        {
            "id": "t-maintenance-window",
            "title": "Maintenance Window",
            "scheduled_time": "2026-01-18T02:00:00Z",
            "recurrence": "weekly",
        },
        {
            "id": "t-traffic-peak",
            "title": "Expected Traffic Peak",
            "scheduled_time": "2026-01-18T12:00:00Z",
            "recurrence": "daily",
        },
    ],
    pattern_types=[
        "Traffic patterns (peaks, valleys, anomalies)",
        "Error cascades (service dependencies)",
        "Latency patterns (slow queries, timeouts)",
        "Deployment impacts (before/after comparisons)",
        "Resource utilization cycles",
    ],
    hierarchy_examples=[
        {
            "level1": "Slow DB Query",
            "level2": "Database Latency",
            "level3": "Performance Degradation",
        },
        {"level1": "Payment Timeout", "level2": "Service Error", "level3": "System Health Issue"},
    ],
    concept_guidelines=(
        "Identify traffic patterns (daily peaks, anomalies)",
        "Detect error correlations across services",
        "Recognize latency patterns by service and operation",
        "Track deployment impacts on system behavior",
        "Notice resource utilization patterns",
    ),
    action_guidelines=(
        "Suggest scaling actions before predicted traffic peaks",
        "Create investigation tickets for recurring errors",
        "Recommend cache optimization for slow queries",
        "Alert on unusual patterns or anomalies",
    ),
    time_guidelines=(
        "Track maintenance windows and deployment schedules",
        "Identify recurring peak traffic times",
        "Create alerts for SLA deadline approaches",
    ),
)


WIKI_DOMAIN = DomainConfig(
    name="wiki",
    description="Wiki pages, notes, and documents ingested as chunked text events",
    event_description="document chunks extracted from wiki pages (markdown), PDFs, and text files",
    node_type_descriptions={
        "event": "A chunk of a document (a slice of text with source metadata)",
        "concept": "A theme/topic that emerges across multiple chunks and documents",
        "action": "Proactive suggestions (e.g., summarize a topic, write follow-up notes, create a TODO)",
        "time": "Optional temporal anchors (generally less relevant for wiki ingestion)",
    },
    concept_examples=[
        {
            "concept_id": "c-topic-vector-databases",
            "title": "Vector Databases",
            "description": "Recurring discussion of vector stores, ANN indexes, and embeddings",
            "strength": 0.6,
            "evidence_count": 6,
        },
        {
            "concept_id": "c-topic-llm-prompting",
            "title": "Prompt Engineering",
            "description": "Repeated notes on prompting patterns, evaluation, and tool use",
            "strength": 0.7,
            "evidence_count": 9,
        },
    ],
    action_examples=[
        {
            "action_id": "a-summarize-topic",
            "title": "Summarize key takeaways on Vector Databases",
            "description": "Produce a short synthesis based on multiple related wiki chunks",
            "priority": "medium",
            "pattern_source": "c-topic-vector-databases",
        }
    ],
    time_examples=[],
    pattern_types=[
        "Topic recurrence across documents",
        "Concept clusters and subtopics",
        "Definitions, examples, and recurring terminology",
        "Contradictions or evolving viewpoints",
    ],
    hierarchy_examples=[
        {
            "level1": "HNSW Index",
            "level2": "Approximate Nearest Neighbor Search",
            "level3": "Vector Retrieval Systems",
        }
    ],
    concept_guidelines=(
        "Create concepts when multiple chunks share a clear theme or recurring terminology",
        "Prefer updating/strengthening existing concepts over creating duplicates",
        "Use hierarchical concepts to group related subtopics (Level 1 specific, Level 2 category, Level 3 abstract)",
        "Ground concepts in multiple chunks when possible (2+ evidence is preferred)",
        "Use MERGE_NODES to consolidate overlapping concepts from different documents",
    ),
    action_guidelines=(
        "Create actions for synthesis: summarize, compare, extract definitions, or draft follow-up notes",
        "Link actions to the concept they derive from via pattern_source",
        "Avoid creating actions for every chunk; prefer actions for strong or recurring concepts",
    ),
    time_guidelines=(),
)


# LoCoMo Domain (Long-term Conversational Memory)
LOCOMO_DOMAIN = DomainConfig(
    name="locomo",
    description="Long-term conversational memory between two speakers",
    event_description="conversation turns between two speakers (User1 and User2)",
    node_type_descriptions={
        "event": "A single conversation turn with speaker attribution and timestamp",
        "concept": "Extracted facts about speakers, their relationships, shared history, and discussed topics",
        "intent": "Goals or desires mentioned in conversation (generally disabled for benchmark)",
        "action": "Legacy alias for intent (disabled for benchmark)",
        "time": "Temporal anchors for when events or facts were mentioned",
    },
    concept_examples=[
        {
            "concept_id": "c-user1-occupation",
            "title": "User1 Occupation",
            "description": "User1 works as a software engineer at Google",
            "strength": 0.9,
            "evidence_count": 2,
        },
        {
            "concept_id": "c-shared-hiking-trip",
            "title": "Hiking Trip Together",
            "description": "User1 and User2 went hiking in Yosemite in June 2023",
            "strength": 0.85,
            "evidence_count": 3,
        },
        {
            "concept_id": "c-speaker-relationship",
            "title": "Speaker Relationship",
            "description": "User1 and User2 are college friends who graduated together in 2018",
            "strength": 0.9,
            "evidence_count": 2,
        },
        {
            "concept_id": "c-user2-preference-coffee",
            "title": "User2 Coffee Preference",
            "description": "User2 prefers oat milk lattes from the local coffee shop",
            "strength": 0.8,
            "evidence_count": 1,
        },
    ],
    action_examples=[],  # Disabled for benchmark
    time_examples=[
        {
            "id": "t-graduation-2018",
            "title": "College Graduation",
            "scheduled_time": "2018-05-15T00:00:00Z",
        },
        {
            "id": "t-hiking-trip-june",
            "title": "Yosemite Hiking Trip",
            "scheduled_time": "2023-06-10T00:00:00Z",
        },
    ],
    pattern_types=[
        "Speaker biographical facts (job, education, location)",
        "Speaker preferences and habits",
        "Shared experiences and history between speakers",
        "Relationship dynamics and how speakers know each other",
        "Topic evolution across multiple sessions",
        "Temporal sequences of events discussed",
    ],
    hierarchy_examples=[
        {
            "level1": "User1 is a software engineer",
            "level2": "User1 Career",
            "level3": "User1 Profile",
        },
        {
            "level1": "Hiking in Yosemite",
            "level2": "Outdoor Activities Together",
            "level3": "Shared Experiences",
        },
    ],
    concept_guidelines=(
        "EXTRACT facts about EACH speaker separately - distinguish User1 from User2",
        "Track speaker relationships (friends, colleagues, family, how they met)",
        "Record shared history and past experiences between speakers",
        "Capture preferences, habits, and biographical details for each speaker",
        "Note temporal context (when facts were mentioned, when events occurred)",
        "UPDATE existing concepts when new information refines or contradicts old info",
        "Track conversation topics that span multiple sessions",
        "Pay attention to names, places, organizations, and specific entities mentioned",
    ),
    action_guidelines=(
        "Avoid creating actions/intents - focus purely on memory extraction for benchmark",
    ),
    time_guidelines=(
        "Create TIME nodes for specific dates mentioned in conversation",
        "Track relative times anchored to session timestamps (e.g., 'next month')",
        "Record event timelines discussed by speakers",
    ),
    disabled_sections=frozenset({"intents"}),
)


# FutureX Domain (Future Event Prediction)
FUTUREX_DOMAIN = DomainConfig(
    name="futurex",
    description="Future event prediction benchmark with multi-step research reasoning",
    event_description="research observations, web search results, and evidence gathered for prediction tasks",
    node_type_descriptions={
        "event": "A piece of evidence or observation gathered during research (search result, article, data point)",
        "concept": "A factual claim, trend, or pattern extracted from research evidence with source attribution",
        "intent": "Research goals or prediction hypotheses to investigate",
        "action": "Legacy alias for intent",
        "time": "Temporal anchors for resolution dates, deadlines, and event schedules",
    },
    concept_examples=[
        {
            "concept_id": "c-evidence-for-yes",
            "title": "Evidence Supporting Yes Outcome",
            "description": "Multiple polls show 65% support for the referendum",
            "strength": 0.75,
            "evidence_count": 3,
            "source": "reuters.com, bbc.com",
        },
        {
            "concept_id": "c-evidence-against-yes",
            "title": "Evidence Against Yes Outcome",
            "description": "Opposition coalition announced boycott, voter turnout may be low",
            "strength": 0.6,
            "evidence_count": 2,
            "source": "aljazeera.com",
        },
        {
            "concept_id": "c-team-form",
            "title": "Team Current Form",
            "description": "Brighton has won 4 of last 5 home matches",
            "strength": 0.8,
            "evidence_count": 5,
            "source": "espn.com, premierleague.com",
        },
        {
            "concept_id": "c-market-trend",
            "title": "Market Trend Analysis",
            "description": "S&P 500 has been above 5000 for 6 consecutive months",
            "strength": 0.85,
            "evidence_count": 4,
            "source": "bloomberg.com, yahoo-finance",
        },
    ],
    action_examples=[
        {
            "action_id": "a-search-more-evidence",
            "title": "Search for more recent news",
            "description": "Need more current information on the topic",
            "priority": "high",
        },
        {
            "action_id": "a-verify-claim",
            "title": "Verify conflicting claims",
            "description": "Two sources disagree, need third source for verification",
            "priority": "medium",
        },
    ],
    time_examples=[
        {
            "id": "t-resolution-date",
            "title": "Prediction Resolution Date",
            "scheduled_time": "2025-09-21T00:00:00Z",
        },
        {
            "id": "t-event-date",
            "title": "Match/Event Date",
            "scheduled_time": "2025-09-20T15:00:00Z",
        },
    ],
    pattern_types=[
        "Evidence for/against prediction outcomes",
        "Source reliability patterns",
        "Temporal relevance (recency of information)",
        "Conflicting claims requiring verification",
        "Expert opinion vs general consensus",
        "Historical patterns for similar events",
    ],
    hierarchy_examples=[
        {
            "level1": "Team Won Last Match",
            "level2": "Recent Team Performance",
            "level3": "Sports Prediction Evidence",
        },
        {
            "level1": "Poll Shows 65% Support",
            "level2": "Public Opinion Trends",
            "level3": "Political Prediction Evidence",
        },
        {
            "level1": "Q3 Earnings Beat Estimates",
            "level2": "Corporate Financial Health",
            "level3": "Economic Prediction Evidence",
        },
    ],
    concept_guidelines=(
        "ALWAYS attribute evidence to sources (URLs, publication names)",
        "Track evidence FOR and AGAINST each prediction outcome separately",
        "Distinguish between hard facts vs opinions vs speculation",
        "Weight recent information higher than older information",
        "Note confidence levels based on source reliability",
        "Identify conflicting claims that need resolution",
        "Create separate concepts for each distinct piece of evidence",
        "Group evidence by prediction outcome (Yes/No, A/B/C, etc.)",
    ),
    action_guidelines=(
        "Suggest searching for more evidence when current evidence is insufficient",
        "Recommend verifying conflicting claims with additional sources",
        "Prioritize research actions based on evidence gaps",
    ),
    time_guidelines=(
        "Create TIME nodes for prediction resolution dates",
        "Track when events are scheduled to occur",
        "Note publication dates of evidence sources",
        "Consider temporal relevance - recent info is more valuable",
    ),
)


# LongMemEval Domain
LONGMEMEVAL_DOMAIN = DomainConfig(
    name="longmemeval",
    description="Long-term memory evaluation benchmark for chat assistants",
    event_description="chat messages between user and assistant",
    node_type_descriptions={
        "event": "A single chat turn (user or assistant message)",
        "concept": "A fact, preference, or piece of knowledge extracted from the chat history",
        "action": "Proactive suggestions (less relevant for this benchmark)",
        "time": "Temporal anchors for when information was revealed or events happened",
    },
    concept_examples=[
        {
            "concept_id": "c-user-profile-education",
            "title": "User Education",
            "description": "User has a degree in Business Administration",
            "strength": 0.9,
            "evidence_count": 1,
        },
        {
            "concept_id": "c-user-preference-coffee",
            "title": "Coffee Preference",
            "description": "User prefers oat milk in their coffee",
            "strength": 0.8,
            "evidence_count": 2,
        },
    ],
    action_examples=[],
    time_examples=[
        {
            "id": "t-graduation-date",
            "title": "Graduation Date",
            "scheduled_time": "2023-05-30T00:00:00Z",
        }
    ],
    pattern_types=[
        "User profile facts (name, age, job, education)",
        "User preferences (food, hobbies, style)",
        "Shared history (past discussions, agreed plans)",
        "Temporal sequence of events",
    ],
    hierarchy_examples=[
        {
            "level1": "Business Degree",
            "level2": "Education History",
            "level3": "User Profile",
        }
    ],
    concept_guidelines=(
        "EXTRACT every factual detail about the user (e.g., 'I am a software engineer', 'I like hiking')",
        "Create CONCEPTs for user preferences, habits, and relationships",
        "Capture specific entities mentioned by the user (names, places, organizations)",
        "Update concepts when new information contradicts or refines old information (Knowledge Update)",
        "Pay attention to temporal context (when something happened) and create TIME nodes if relevant",
    ),
    action_guidelines=(
        "Generally avoid creating actions unless the user explicitly requests a reminder or task",
    ),
    time_guidelines=(
        "Create TIME nodes when specific dates or relative times (e.g., 'next week') are mentioned",
    ),
    disabled_sections=frozenset({"intents"}),
)


# Claude Code Domain (AI coding assistant sessions)
CLAUDE_CODE_DOMAIN = DomainConfig(
    name="claude-code",
    description="Claude Code AI coding assistant session events",
    event_description="Claude Code session events (tool invocations, git operations, conversations, errors)",
    node_type_descriptions={
        "event": "Direct representations of session actions (tool calls, git ops, messages, errors)",
        "concept": "Higher-level patterns in coding workflow (e.g., 'Read-Grep-Edit Debug Pattern', 'Test-Driven Development Flow')",
        "intent": "Goals or tasks being pursued (e.g., 'Fix failing test suite', 'Refactor duplicated logic')",
        "action": "Goals or tasks being pursued (legacy alias for 'intent')",
        "time": "Temporal anchors for session phases, deadlines, and productivity windows",
    },
    concept_examples=[
        {
            "concept_id": "c-debug-pattern",
            "title": "Read-Grep-Edit Debug Pattern",
            "description": "Recurring sequence: Read file, Grep for references, Edit to fix, Bash to test",
            "strength": 0.8,
            "evidence_count": 6,
        },
        {
            "concept_id": "c-tdd-flow",
            "title": "Test-Driven Development Flow",
            "description": "Pattern of writing tests first, running them to fail, then implementing until green",
            "strength": 0.7,
            "evidence_count": 4,
        },
        {
            "concept_id": "c-codebase-understanding",
            "title": "Codebase Architecture Understanding",
            "description": "Progressive exploration via Glob/Read/Grep building mental model of project structure",
            "strength": 0.6,
            "evidence_count": 8,
        },
        {
            "concept_id": "c-incremental-commit",
            "title": "Incremental Commit Strategy",
            "description": "Small, focused commits after each logical unit of work",
            "strength": 0.75,
            "evidence_count": 5,
        },
    ],
    action_examples=[
        {
            "action_id": "a-fix-tests",
            "title": "Fix failing test suite",
            "description": "Multiple test failures detected, need systematic debugging",
            "priority": "high",
            "pattern_source": "c-debug-pattern",
        },
        {
            "action_id": "a-refactor-duplication",
            "title": "Refactor duplicated logic",
            "description": "Same pattern appears in multiple files, extract shared utility",
            "priority": "medium",
            "pattern_source": "c-codebase-understanding",
        },
    ],
    time_examples=[
        {
            "id": "t-session-start",
            "title": "Session Start",
            "scheduled_time": "2026-01-18T09:00:00Z",
        },
        {
            "id": "t-deep-work-window",
            "title": "Deep Implementation Phase",
            "scheduled_time": "2026-01-18T10:00:00Z",
            "recurrence": "per-session",
        },
    ],
    pattern_types=[
        "Tool usage sequences (Read→Grep→Edit→Bash cycles)",
        "Debug strategies (hypothesis-test-fix loops)",
        "Codebase exploration patterns (breadth-first vs targeted search)",
        "Human-AI interaction rhythms (instruction→execution→correction)",
        "Session productivity phases (exploration, implementation, testing, cleanup)",
        "Error recovery patterns (retry strategies, alternative approaches)",
    ],
    hierarchy_examples=[
        {
            "level1": "Grep for Function Definition",
            "level2": "Codebase Exploration",
            "level3": "Architecture Understanding",
        },
        {
            "level1": "Run pytest",
            "level2": "Test Verification",
            "level3": "Quality Assurance Flow",
        },
    ],
    concept_guidelines=(
        "Identify tool usage sequences that form recognizable workflows (e.g., Read→Edit→Bash test cycles)",
        "Detect debug strategies: how errors are diagnosed and resolved across multiple tool calls",
        "Recognize codebase exploration patterns (Glob/Grep/Read sequences building understanding)",
        "Track human-AI interaction patterns (when corrections happen, what triggers re-planning)",
        "Notice session productivity phases (ramp-up, deep work, wind-down)",
        "Identify knowledge accumulation — which files/modules become well-understood over time",
    ),
    action_guidelines=(
        "Suggest next logical steps based on current workflow phase",
        "Identify when a different debugging strategy might be more effective",
        "Recommend commits at natural breakpoints",
        "Flag potential test gaps after implementation changes",
    ),
    time_guidelines=(
        "Track session start/end for productivity analysis",
        "Identify phase transitions within sessions (exploration → implementation → testing)",
        "Recognize time-of-day patterns in session effectiveness",
    ),
    extra_sections={
        "claude_code.tool_context": """## Claude Code Tool Taxonomy

Claude Code operates through discrete tool invocations. Understanding the tool types helps
identify workflow patterns:

**File Operations**: Read (view files), Edit (modify files), Write (create files), Glob (find files by pattern), Grep (search file contents)
**Execution**: Bash (run shell commands — tests, builds, git, etc.)
**Web**: WebSearch (search the web), WebFetch (fetch URL content)
**Git Operations**: Expressed as Bash calls — git commit, git push, git branch, gh pr create, etc.
**Conversation**: Human messages (instructions, corrections, approvals) and Claude responses (explanations, plans, questions)

**Key Patterns to Watch For:**
- Read→Grep→Edit cycles indicate targeted bug fixes
- Glob→Read→Read sequences indicate codebase exploration
- Bash(test)→Edit→Bash(test) loops indicate test-driven development
- Conversation corrections after tool failures indicate learning moments
- Git commit frequency indicates work granularity style
""",
    },
    extra_section_position="after_tools",
)


# MSC Domain (Multi-Session Chat)
MSC_DOMAIN = DomainConfig(
    name="msc",
    description="Multi-session chat benchmark for persona consistency and fact recall",
    event_description="conversation turns between two speakers across multiple chat sessions",
    node_type_descriptions={
        "event": "A single conversation turn with speaker attribution",
        "concept": "Persona facts, preferences, and biographical details per speaker",
        "action": "Disabled for this benchmark",
        "time": "Temporal anchors for session boundaries and mentioned dates",
    },
    concept_examples=[
        {
            "concept_id": "c-speaker1-hobby",
            "title": "Speaker 1 Hobby",
            "description": "Speaker 1 enjoys hiking in the mountains on weekends",
            "strength": 0.85,
            "evidence_count": 2,
        },
        {
            "concept_id": "c-speaker2-job",
            "title": "Speaker 2 Occupation",
            "description": "Speaker 2 works as a nurse at a local hospital",
            "strength": 0.9,
            "evidence_count": 1,
        },
        {
            "concept_id": "c-speaker1-pet",
            "title": "Speaker 1 Pet",
            "description": "Speaker 1 owns a golden retriever named Max",
            "strength": 0.8,
            "evidence_count": 3,
        },
    ],
    action_examples=[],
    time_examples=[
        {
            "id": "t-session-boundary",
            "title": "Session 2 Start",
            "scheduled_time": "2024-01-02T10:00:00Z",
        }
    ],
    pattern_types=[
        "Speaker biographical facts (job, education, location, family)",
        "Speaker preferences and habits (food, hobbies, music)",
        "Persona consistency across sessions",
        "Fact recall from earlier sessions",
        "Topic evolution across sessions",
    ],
    hierarchy_examples=[
        {
            "level1": "Speaker 1 owns a dog named Max",
            "level2": "Speaker 1 Pets",
            "level3": "Speaker 1 Profile",
        },
        {
            "level1": "Both enjoy Italian food",
            "level2": "Shared Preferences",
            "level3": "Speaker Relationship",
        },
    ],
    concept_guidelines=(
        "EXTRACT persona facts for EACH speaker separately - distinguish Speaker 1 from Speaker 2",
        "Track biographical details: occupation, hobbies, family, pets, location",
        "Record preferences: food, music, activities, likes and dislikes",
        "UPDATE existing concepts when speakers elaborate on previously mentioned facts",
        "Pay special attention to facts that carry across sessions - these test long-term memory",
        "Note contradictions or updates to previously stated facts",
        "Capture specific names, places, and entities mentioned by each speaker",
    ),
    action_guidelines=("Avoid creating actions - focus on memory extraction for benchmark",),
    time_guidelines=("Create TIME nodes only for explicitly mentioned dates or events",),
    disabled_sections=frozenset({"intents"}),
)


# BABILong Domain (Logic Reasoning with Noise)
BABILONG_DOMAIN = DomainConfig(
    name="babilong",
    description="Logic reasoning tasks with facts hidden in noisy long contexts",
    event_description="narrative statements about entities, their locations, and possessions",
    node_type_descriptions={
        "event": "A statement about an entity action (movement, pickup, drop, handoff)",
        "concept": "Current state of an entity (location, possessions, counts)",
        "action": "Disabled for this benchmark",
        "time": "Sequence ordering of entity state changes",
    },
    concept_examples=[
        {
            "concept_id": "c-john-location",
            "title": "John's Current Location",
            "description": "John is currently in the kitchen",
            "strength": 1.0,
        },
        {
            "concept_id": "c-mary-possessions",
            "title": "Mary's Possessions",
            "description": "Mary is carrying the apple",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Entity locations (who is where)",
        "Entity possessions (who has what)",
        "Entity movements (who went where)",
        "Object transfers (who gave what to whom)",
    ],
    hierarchy_examples=[
        {
            "level1": "John is in the kitchen",
            "level2": "John's State",
            "level3": "Entity States",
        },
    ],
    concept_guidelines=(
        "Track CURRENT state of each entity (person) - especially their LOCATION",
        "UPDATE (not create new) concept nodes when an entity moves or changes state",
        "Distinguish relevant narrative facts from irrelevant filler/noise text",
        "Ignore sentences that do not involve tracked entities, locations, or objects",
        "Maintain high-confidence edges for causal movement chains",
        "For possession tasks, track who is carrying which objects precisely",
        "For counting tasks, track quantities and list items explicitly",
    ),
    action_guidelines=("Avoid creating actions - focus on entity state tracking",),
    time_guidelines=("Track sequence ordering of entity movements if relevant",),
    disabled_sections=frozenset({"intents"}),
)


# ---------------------------------------------------------------------------
# Benchmark domains: Multi-Hop Reasoning
# ---------------------------------------------------------------------------

MUTUAL_DOMAIN = DomainConfig(
    name="mutual",
    description="Multi-turn dialogue reasoning with multiple-choice responses",
    event_description="dialogue turns between two speakers in a social conversation",
    node_type_descriptions={
        "event": "A dialogue turn from one speaker",
        "concept": "Inferred social intent, emotion, or implication from the dialogue",
        "action": "Disabled for this benchmark",
        "time": "Dialogue turn ordering",
    },
    concept_examples=[
        {
            "concept_id": "c-speaker-intent",
            "title": "Speaker B's Intent",
            "description": "Speaker B is politely declining an invitation",
            "strength": 0.9,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Social intentions (accepting, declining, suggesting)",
        "Emotional states (happy, disappointed, surprised)",
        "Implicit meaning (sarcasm, politeness, indirectness)",
    ],
    concept_guidelines=(
        "Extract the implied social intent behind each dialogue turn",
        "Track speaker emotional states and how they change through conversation",
        "Identify indirect speech acts (hints, implications, suggestions)",
    ),
    action_guidelines=("Avoid creating actions - focus on dialogue understanding",),
    time_guidelines=("Track dialogue turn order for context flow",),
    disabled_sections=frozenset({"intents"}),
)

MUSIQUE_DOMAIN = DomainConfig(
    name="musique",
    description="Multi-hop question answering requiring chained paragraph reasoning",
    event_description="information paragraphs that may contain facts needed for multi-step reasoning",
    node_type_descriptions={
        "event": "A paragraph containing factual information about entities",
        "concept": "A fact or relationship extracted from a paragraph",
        "action": "Disabled for this benchmark",
        "time": "Disabled for this benchmark",
    },
    concept_examples=[
        {
            "concept_id": "c-director-film",
            "title": "Film Director",
            "description": "Christopher Nolan directed Inception",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Entity relationships (who directed what, who married whom)",
        "Factual attributes (birthplace, occupation, affiliation)",
        "Chains of inference (A->B->C multi-hop)",
    ],
    concept_guidelines=(
        "Extract key entity-relation-entity triples from each paragraph",
        "Create edges connecting related concepts across paragraphs",
        "Distinguish supporting paragraphs from distractors",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Not applicable",),
    disabled_sections=frozenset({"intents"}),
)


# ---------------------------------------------------------------------------
# Benchmark domains: Streaming & Conflicts
# ---------------------------------------------------------------------------

STREAMINGQA_DOMAIN = DomainConfig(
    name="streamingqa",
    description="Temporal knowledge QA where facts change over time",
    event_description="news articles with publication dates containing evolving factual information",
    node_type_descriptions={
        "event": "A news article or passage with a specific publication date",
        "concept": "A factual claim that may be updated or superseded over time",
        "action": "Disabled for this benchmark",
        "time": "Publication date or knowledge validity period",
    },
    concept_examples=[
        {
            "concept_id": "c-ceo-company",
            "title": "CEO of Company X",
            "description": "John is the current CEO of Company X (as of 2020-01)",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Evolving facts (CEO changes, company mergers)",
        "Temporal validity (fact true from date X to date Y)",
        "Knowledge updates (old answer superseded by new one)",
    ],
    concept_guidelines=(
        "Track temporal validity of each fact - when it became true and if it was superseded",
        "UPDATE concept nodes when new information contradicts old facts",
        "Maintain version history via edges to preserve old knowledge",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=(
        "Create TIME nodes for publication dates and fact validity periods",
        "Use DEADLINE_FOR edges to mark fact expiration",
    ),
    disabled_sections=frozenset({"intents"}),
)

RGB_DOMAIN = DomainConfig(
    name="rgb",
    description="Robustness benchmark testing noise filtering and conflict resolution",
    event_description="passages containing factual information, noise, or conflicting claims",
    node_type_descriptions={
        "event": "A text passage that may contain facts, noise, or contradictions",
        "concept": "A verified or contested factual claim",
        "action": "Disabled for this benchmark",
        "time": "Disabled for this benchmark",
    },
    concept_examples=[
        {
            "concept_id": "c-verified-fact",
            "title": "Capital of France",
            "description": "Paris is the capital of France (high confidence)",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Verified facts vs noise",
        "Conflicting claims requiring resolution",
        "Counterfactual information that should be rejected",
    ],
    concept_guidelines=(
        "Distinguish factual content from noise or irrelevant information",
        "When encountering conflicting claims, track both and note the conflict",
        "Assign lower strength to claims that contradict established facts",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Not applicable",),
    disabled_sections=frozenset({"intents"}),
)

TIMEQA_DOMAIN = DomainConfig(
    name="timeqa",
    description="Time-sensitive question answering with temporal constraints",
    event_description="passages about entities with time-evolving attributes and states",
    node_type_descriptions={
        "event": "A passage describing entity states at specific time periods",
        "concept": "A temporally-bound fact (who held role X during period Y)",
        "action": "Disabled for this benchmark",
        "time": "Temporal boundaries for fact validity (start/end dates)",
    },
    concept_examples=[
        {
            "concept_id": "c-role-period",
            "title": "CEO tenure",
            "description": "John was CEO of Company X from 2015 to 2019",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Entity roles with temporal boundaries",
        "State transitions (before/after events)",
        "Overlapping time periods",
    ],
    concept_guidelines=(
        "Extract temporal bounds for every fact (start date, end date if known)",
        "Create TIME nodes for explicit date references",
        "Track state transitions: who/what changed when",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=(
        "Create TIME nodes for all date references",
        "Use DEADLINE_FOR edges to link time constraints to facts",
    ),
    disabled_sections=frozenset({"intents"}),
)


# ---------------------------------------------------------------------------
# Benchmark domains: Long-Form Narrative
# ---------------------------------------------------------------------------

NARRATIVEQA_DOMAIN = DomainConfig(
    name="narrativeqa",
    description="Question answering on full books and movie scripts",
    event_description="narrative segments from books or movie scripts, fed as a chapter stream",
    node_type_descriptions={
        "event": "A narrative segment (chapter excerpt, scene, or passage)",
        "concept": "A character trait, plot point, theme, or relationship",
        "action": "Disabled for this benchmark",
        "time": "Narrative chronology and plot progression",
    },
    concept_examples=[
        {
            "concept_id": "c-character-trait",
            "title": "Protagonist's Motivation",
            "description": "The protagonist seeks redemption after a past mistake",
            "strength": 0.9,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Character attributes and development",
        "Plot events and causal chains",
        "Themes and motifs",
        "Inter-character relationships",
    ],
    concept_guidelines=(
        "Extract character traits, motivations, and relationships",
        "Track major plot events and their consequences",
        "Identify recurring themes and motifs",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Track narrative chronology if relevant to plot",),
    disabled_sections=frozenset({"intents"}),
)

QMSUM_DOMAIN = DomainConfig(
    name="qmsum",
    description="Query-based meeting summarization from multi-speaker transcripts",
    event_description="meeting transcript turns from multiple speakers discussing topics",
    node_type_descriptions={
        "event": "A speaker utterance in a meeting transcript",
        "concept": "A discussion topic, decision, action item, or key point",
        "action": "Disabled for this benchmark",
        "time": "Meeting progression and topic transitions",
    },
    concept_examples=[
        {
            "concept_id": "c-decision",
            "title": "Timeline Decision",
            "description": "The team agreed on a 2-week timeline for the feature",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Decisions and agreements",
        "Action items and assignments",
        "Topic discussions and conclusions",
        "Speaker opinions and disagreements",
    ],
    concept_guidelines=(
        "Extract key decisions, action items, and discussion outcomes",
        "Track which speaker made which points",
        "Identify topic transitions in the meeting flow",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Track meeting flow and topic progression",),
    disabled_sections=frozenset({"intents"}),
)


# ---------------------------------------------------------------------------
# Benchmark domains: Proactive & Commonsense
# ---------------------------------------------------------------------------

SOCIALIQA_DOMAIN = DomainConfig(
    name="socialiqa",
    description="Social commonsense reasoning about human interactions and intent",
    event_description="social situations describing human interactions and their consequences",
    node_type_descriptions={
        "event": "A social situation or scenario involving human actors",
        "concept": "An inferred emotional state, social norm, or behavioral expectation",
        "action": "Disabled for this benchmark",
        "time": "Disabled for this benchmark",
    },
    concept_examples=[
        {
            "concept_id": "c-emotion",
            "title": "Emotional Response",
            "description": "Sydney felt guilty for not being able to help",
            "strength": 0.8,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Emotional reactions to events",
        "Social norms and expectations",
        "Cause-effect in social interactions",
    ],
    concept_guidelines=(
        "Infer emotional states of actors in the situation",
        "Identify social norms being followed or violated",
        "Track cause-effect chains in social interactions",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Not applicable",),
    disabled_sections=frozenset({"intents"}),
)

TOMI_DOMAIN = DomainConfig(
    name="tomi",
    description="Theory of Mind - tracking what different agents know and believe",
    event_description="stories about agent movements and object interactions with perspective tracking",
    node_type_descriptions={
        "event": "An agent action (entering, leaving, moving objects)",
        "concept": "An agent's belief about the world state (may differ from reality)",
        "action": "Disabled for this benchmark",
        "time": "Sequence of actions determining who saw what",
    },
    concept_examples=[
        {
            "concept_id": "c-belief",
            "title": "Sally's Belief about Ball",
            "description": "Sally believes the ball is in the basket (hasn't seen it moved)",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Agent locations (who is where)",
        "Object locations (reality vs beliefs)",
        "Agent beliefs (what each agent thinks is true)",
        "Observation tracking (who saw what happen)",
    ],
    concept_guidelines=(
        "Track each agent's BELIEF about object locations separately from reality",
        "When an agent leaves a room, they stop observing changes",
        "An agent's belief reflects the last state they observed",
        "Distinguish 1st-order beliefs (what X thinks) from 2nd-order (what X thinks Y thinks)",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Track action sequence to determine observation windows",),
    disabled_sections=frozenset({"intents"}),
)

SAFETYBENCH_DOMAIN = DomainConfig(
    name="safetybench",
    description="Safety evaluation benchmark for ethical reasoning and harm prevention",
    event_description="safety-related scenarios requiring ethical judgment",
    node_type_descriptions={
        "event": "A safety scenario or ethical question",
        "concept": "A safety principle, ethical norm, or risk assessment",
        "action": "Disabled for this benchmark",
        "time": "Disabled for this benchmark",
    },
    concept_examples=[
        {
            "concept_id": "c-safety-principle",
            "title": "Privacy Rights",
            "description": "User data collection requires explicit consent",
            "strength": 1.0,
        },
    ],
    action_examples=[],
    time_examples=[],
    pattern_types=[
        "Ethical principles and norms",
        "Safety risks and mitigations",
        "Legal and regulatory constraints",
    ],
    concept_guidelines=(
        "Identify the core ethical principle at stake in each scenario",
        "Track safety category (ethics, health, legal, bias, etc.)",
        "Distinguish safe from harmful options",
    ),
    action_guidelines=("Avoid creating actions",),
    time_guidelines=("Not applicable",),
    disabled_sections=frozenset({"intents"}),
)


# Learning Domain (NeoLearn — textbook/article ingestion for study)
LEARNING_DOMAIN = DomainConfig(
    name="learning",
    description="Document chunks from textbooks, articles, and study materials ingested for learning",
    event_description="document chunks from textbooks, articles, and study materials ingested for learning",
    node_type_descriptions={
        "event": "A chunk of a study document (a slice of text with source metadata)",
        "concept": "A key learning topic, theme, or knowledge point that emerges across chunks",
        "action": "Disabled for learning (scheduling is handled by the NeoLearn backend)",
        "time": "Optional temporal anchors (generally less relevant for learning ingestion)",
    },
    concept_examples=[
        {
            "concept_id": "c-topic-mitosis",
            "title": "Mitosis",
            "description": "The process of cell division producing two identical daughter cells",
            "strength": 0.7,
            "evidence_count": 4,
        },
        {
            "concept_id": "c-topic-supply-demand",
            "title": "Supply and Demand",
            "description": "Economic model describing how prices are determined in a market",
            "strength": 0.8,
            "evidence_count": 6,
        },
        {
            "concept_id": "c-topic-jiqixuexi",
            "title": "\u673a\u5668\u5b66\u4e60",
            "description": "\u7814\u7a76\u8ba1\u7b97\u673a\u5982\u4f55\u4ece\u6570\u636e\u4e2d\u81ea\u52a8\u5b66\u4e60\u548c\u6539\u8fdb\u7684\u4eba\u5de5\u667a\u80fd\u5206\u652f",
            "strength": 0.8,
            "evidence_count": 5,
        },
    ],
    action_examples=[
        {
            "intent_id": "int-review-mitosis",
            "title": "Review Mitosis Process",
            "description": "Review the stages of mitosis \u2014 prophase, metaphase, anaphase, telophase",
            "priority": "medium",
            "status": "pending",
            "pattern_source": "c-topic-mitosis",
            "suggested_time": "tomorrow morning",
        },
        {
            "intent_id": "int-practice-supply-demand",
            "title": "Practice Supply-Demand Graphs",
            "description": "Draw and analyze supply-demand equilibrium graphs",
            "priority": "high",
            "status": "pending",
            "pattern_source": "c-topic-supply-demand",
        },
        {
            "intent_id": "int-review-jiqixuexi",
            "title": "\u590d\u4e60\u673a\u5668\u5b66\u4e60\u57fa\u7840",
            "description": "\u56de\u987e\u76d1\u7763\u5b66\u4e60\u3001\u65e0\u76d1\u7763\u5b66\u4e60\u548c\u5f3a\u5316\u5b66\u4e60\u7684\u533a\u522b",
            "priority": "medium",
            "status": "pending",
            "pattern_source": "c-topic-jiqixuexi",
        },
    ],
    time_examples=[],
    pattern_types=[
        "Key definitions and terminology",
        "Cause-effect relationships",
        "Concept clusters and subtopics",
        "Prerequisite relationships between topics",
        "Historical events and figures",
        "Recurring themes across chapters or documents",
    ],
    hierarchy_examples=[
        {
            "level1": "Mitosis",
            "level2": "Cell Division",
            "level3": "Cell Biology",
        },
        {
            "level1": "Supply and Demand",
            "level2": "Market Equilibrium",
            "level3": "Microeconomics",
        },
    ],
    concept_guidelines=(
        "Extract learnable concepts: key terms, definitions, theorems, and principles",
        "Identify cause-effect relationships and prerequisite dependencies between topics",
        "Create hierarchical concepts (specific → category → broad field)",
        "Prefer updating/strengthening existing concepts over creating duplicates",
        "Ground concepts in multiple chunks when possible (2+ evidence is preferred)",
        "Use MERGE_NODES to consolidate overlapping concepts from different documents",
        "Capture named entities: people, places, dates, formulas that are study-worthy",
        "When processing non-English documents, generate concept titles and descriptions in the document's language",
    ),
    action_guidelines=(
        "Create INTENT nodes for learning goals that emerge from document analysis",
        "When a concept cluster forms (3+ related concepts), create a 'review' intent",
        "When prerequisite gaps are detected, create a 'study prerequisite' intent",
        "Include suggested_time based on concept complexity (simple=10min, complex=30min)",
        "Link intents to the source concepts they target",
        "Set priority: 'high' for foundational/prerequisite concepts, 'medium' for details",
    ),
    time_guidelines=(),
)


# Registry of available domains
DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "personal-timeline": PERSONAL_TIMELINE_DOMAIN,
    "computer-activity": COMPUTER_ACTIVITY_DOMAIN,
    "wiki": WIKI_DOMAIN,
    "service-logs": SERVICE_LOGS_DOMAIN,
    "locomo": LOCOMO_DOMAIN,
    "longmemeval": LONGMEMEVAL_DOMAIN,
    "futurex": FUTUREX_DOMAIN,
    "claude-code": CLAUDE_CODE_DOMAIN,
    "msc": MSC_DOMAIN,
    "babilong": BABILONG_DOMAIN,
    "mutual": MUTUAL_DOMAIN,
    "musique": MUSIQUE_DOMAIN,
    "streamingqa": STREAMINGQA_DOMAIN,
    "rgb": RGB_DOMAIN,
    "timeqa": TIMEQA_DOMAIN,
    "narrativeqa": NARRATIVEQA_DOMAIN,
    "qmsum": QMSUM_DOMAIN,
    "socialiqa": SOCIALIQA_DOMAIN,
    "tomi": TOMI_DOMAIN,
    "safetybench": SAFETYBENCH_DOMAIN,
    "learning": LEARNING_DOMAIN,
}


def get_domain_config(name: str) -> DomainConfig:
    """Get a domain configuration by name.

    Args:
        name: Domain name (e.g., "personal-timeline").

    Returns:
        The domain configuration.

    Raises:
        KeyError: If domain not found.
    """
    if name not in DOMAIN_REGISTRY:
        available = ", ".join(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain: {name}. Available: {available}")
    return DOMAIN_REGISTRY[name]


def register_domain(config: DomainConfig) -> None:
    """Register a new domain configuration.

    Args:
        config: Domain configuration to register.
    """
    DOMAIN_REGISTRY[config.name] = config
