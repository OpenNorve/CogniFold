"""Plan execution with atomicity guarantees."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.graph.validator import ValidationReport
    from cognifold.models.plan import Operation, UpdatePlan

logger = logging.getLogger(__name__)


def _is_near_duplicate(graph: ConceptGraph, new_title: str, node_type: str) -> str | None:
    """Check if a near-duplicate concept already exists.

    Only applies to ``"concept"`` nodes.  Compares the normalized (lowercase,
    stripped) title against existing concept titles using exact match and
    simple containment heuristics.

    Args:
        graph: The concept graph to search.
        new_title: Title of the candidate new node.
        node_type: Node type string (only ``"concept"`` is checked).

    Returns:
        The existing node ID if a near-duplicate is found, otherwise ``None``.
    """
    if node_type != "concept":
        return None
    from cognifold.models.node import NodeType

    normalized = new_title.strip().lower()
    if not normalized:
        return None
    for node in graph.get_nodes_by_type(NodeType.CONCEPT):
        existing = (node.data.get("title") or "").strip().lower()
        if not existing:
            continue
        if existing == normalized:
            return node.id
        # Simple containment check for near-duplicates
        if (
            len(normalized) > 5
            and len(existing) > 5
            and (normalized in existing or existing in normalized)
        ):
            return node.id
    return None


@dataclass
class ExecutionResult:
    """Result of executing an UpdatePlan."""

    success: bool
    plan_id: str
    operations_completed: int = 0
    error: str | None = None
    failed_at_operation: int | None = None
    execution_time_ms: float = 0.0
    validation_report: ValidationReport | None = None

    @property
    def has_integrity_issues(self) -> bool:
        """Check if validation found any integrity issues."""
        if self.validation_report is None:
            return False
        return not self.validation_report.is_valid


@dataclass
class GraphSnapshot:
    """Snapshot of graph state for rollback."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


class PlanExecutor:
    """Executes UpdatePlans against the ConceptGraph.

    Provides atomic execution with rollback on failure.
    Optionally validates graph integrity after execution.
    """

    def __init__(
        self,
        graph: ConceptGraph,
        validate_after_execution: bool = False,
        log_integrity_issues: bool = True,
        skip_embedding: bool = False,
        graph_sync: Any = None,
    ):
        """Initialize with a reference to the graph.

        Args:
            graph: The graph to execute plans against.
            validate_after_execution: If True, run GraphValidator after each execution.
            log_integrity_issues: If True, log any integrity issues found.
            skip_embedding: If True, skip per-node embedding API calls (for fast ingest).
            graph_sync: Optional GraphSyncWriter for write-through to Supabase.
        """
        self._graph = graph
        self._validate_after_execution = validate_after_execution
        self._log_integrity_issues = log_integrity_issues
        self._skip_embedding = skip_embedding
        self._graph_sync = graph_sync

    def execute(self, plan: UpdatePlan) -> ExecutionResult:
        """Execute an update plan atomically.

        All operations succeed or the graph is rolled back to its
        state before execution began.

        Args:
            plan: The validated update plan.

        Returns:
            ExecutionResult with success status and details.
        """
        start_time = datetime.now()

        # Lazy snapshot — only created if we actually need to rollback.
        snapshot: GraphSnapshot | None = None

        # Sort operations so ADD_NODEs execute before ADD_EDGEs.
        # LLM-generated plans may have edges before their target nodes.
        from cognifold.models.plan import OperationType

        op_order = {
            OperationType.ADD_NODE: 0,
            OperationType.UPDATE_NODE: 1,
            OperationType.ADD_EDGE: 2,
            OperationType.REMOVE_EDGE: 3,
            OperationType.REMOVE_NODE: 4,
            OperationType.MERGE_NODES: 5,
        }
        sorted_ops = sorted(plan.operations, key=lambda o: op_order.get(o.op, 99))

        # Collect all node IDs referenced by edge operations so ADD_NODE
        # can resolve agent-assigned IDs regardless of which data key holds them.
        edge_referenced_ids: set[str] = set()
        for op in plan.operations:
            if op.op in (OperationType.ADD_EDGE, OperationType.REMOVE_EDGE):
                if op.source_id:
                    edge_referenced_ids.add(op.source_id)
                if op.target_id:
                    edge_referenced_ids.add(op.target_id)
        self._edge_referenced_ids = edge_referenced_ids

        operation_index = 0
        try:
            # Create snapshot lazily — only before the first mutation
            snapshot = self._create_snapshot()

            for op in sorted_ops:
                self._execute_operation(op)
                operation_index += 1

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Detect orphan concept/intent nodes (no edges at all)
            self._detect_orphan_nodes(plan.plan_id)

            # Run post-execution validation if enabled
            validation_report = None
            if self._validate_after_execution:
                validation_report = self._validate_graph()
                if self._log_integrity_issues and validation_report:
                    self._log_validation_issues(validation_report, plan.plan_id)

            return ExecutionResult(
                success=True,
                plan_id=plan.plan_id,
                operations_completed=len(plan.operations),
                execution_time_ms=execution_time,
                validation_report=validation_report,
            )

        except Exception as e:
            # Rollback on any failure
            logger.error(
                "Plan %s failed at operation %d: %s — rolling back",
                plan.plan_id,
                operation_index,
                e,
            )
            if snapshot is not None:
                self._restore_snapshot(snapshot)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ExecutionResult(
                success=False,
                plan_id=plan.plan_id,
                operations_completed=operation_index,
                error=str(e),
                failed_at_operation=operation_index,
                execution_time_ms=execution_time,
            )

    def _create_snapshot(self) -> GraphSnapshot:
        """Create a snapshot of the current graph state."""
        nodes = []
        for node in self._graph.get_all_nodes():
            nodes.append(
                {
                    "id": node.id,
                    "type": node.type.value,
                    "data": node.data.copy(),
                    "created_at": node.created_at,
                    "last_accessed": node.last_accessed,
                    "access_count": node.access_count,
                }
            )

        edges = []
        for edge in self._graph.get_all_edges():
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    "created_at": edge.created_at,
                    "metadata": edge.metadata,
                }
            )

        return GraphSnapshot(nodes=nodes, edges=edges)

    def _restore_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Restore the graph to a previous state."""
        from cognifold.models.node import Edge, Node, NodeType

        # Clear current graph
        self._graph.clear()

        # Restore nodes
        for node_data in snapshot.nodes:
            node = Node(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                data=node_data["data"],
                created_at=node_data["created_at"],
                last_accessed=node_data["last_accessed"],
                access_count=node_data["access_count"],
            )
            self._graph.add_node(node)

        # Restore edges
        for edge_data in snapshot.edges:
            edge = Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=edge_data.get("edge_type"),
                weight=edge_data.get("weight", 1.0),
                created_at=edge_data["created_at"],
                metadata=edge_data.get("metadata", {}),
            )
            self._graph.add_edge(edge)

    def _execute_operation(self, op: Operation) -> None:
        """Execute a single operation."""
        from cognifold.models.node import Edge, Node, NodeType
        from cognifold.models.plan import OperationType

        if op.op == OperationType.ADD_NODE:
            node_id = self._resolve_add_node_id(op)

            # Near-duplicate concept detection: convert to UPDATE if duplicate
            new_title = (op.data or {}).get("title", "")
            dup_id = _is_near_duplicate(self._graph, new_title, op.node_type or "")
            if dup_id is not None:
                logger.info(
                    "ADD_NODE dedup: '%s' is near-duplicate of existing '%s' — converting to UPDATE",
                    new_title,
                    dup_id,
                )
                # Reinforce the existing concept instead of creating a duplicate
                update_data: dict[str, Any] = {}
                existing = self._graph.get_node(dup_id)
                old_strength = existing.data.get("strength", 0.5)
                update_data["strength"] = min(old_strength + 0.1, 1.0)
                evidence = existing.data.get("evidence_count", 1)
                update_data["evidence_count"] = evidence + 1
                self._graph.update_node(dup_id, update_data)
                if self._graph_sync is not None:
                    self._graph_sync.on_node_updated(dup_id, update_data, "concept")
                return

            # Generate Embedding (skipped in fast-ingest mode)
            embedding: list[float] | None = None
            if not self._skip_embedding:
                from cognifold.utils.embeddings import get_embedding_service

                text_to_embed = ""
                if op.data:
                    text_to_embed += str(op.data.get("title", "")) + " "
                    text_to_embed += str(op.data.get("description", "")) + " "
                if op.reasoning:
                    text_to_embed += op.reasoning

                embedding = get_embedding_service().embed_text(text_to_embed.strip())

            node = Node(
                id=node_id,
                type=NodeType(op.node_type) if op.node_type else NodeType.EVENT,
                data=op.data or {},
                reasoning=op.reasoning,
                grounded_in=op.grounded_in or [],
                embedding=embedding,
            )
            self._graph.add_node(node)
            if self._graph_sync is not None:
                self._graph_sync.on_node_added(node_id, node.type.value, op.data or {}, embedding)

        elif op.op == OperationType.UPDATE_NODE:
            if op.node_id and op.data:
                # Get existing node to track changes
                existing_node = self._graph.get_node(op.node_id)
                if existing_node and op.update_reasoning:
                    # Track what changed
                    changes = {}
                    for key, new_val in op.data.items():
                        old_val = existing_node.data.get(key)
                        if old_val != new_val:
                            changes[key] = {"old": old_val, "new": new_val}

                    if changes:
                        # Add to update history
                        updated_node = existing_node.add_update_history(
                            update_reasoning=op.update_reasoning,
                            changes=changes,
                        )
                        # First update with history, then apply data changes
                        self._graph.internal_graph.nodes[op.node_id]["node"] = updated_node

                self._graph.update_node(op.node_id, op.data)
                if self._graph_sync is not None:
                    ntype = existing_node.type.value if existing_node else "event"
                    self._graph_sync.on_node_updated(op.node_id, op.data, ntype)

        elif op.op == OperationType.REMOVE_NODE:
            if op.node_id:
                self._graph.remove_node(op.node_id)
                if self._graph_sync is not None:
                    self._graph_sync.on_node_removed(op.node_id)

        elif op.op == OperationType.ADD_EDGE:
            if op.source_id and op.target_id:
                source_id = op.source_id
                target_id = op.target_id

                # Resolve references: LLMs often use titles/names instead of IDs
                if not self._graph.has_node(source_id):
                    resolved = self._resolve_node_ref(source_id)
                    if resolved:
                        source_id = resolved
                    else:
                        logger.warning(
                            "ADD_EDGE skipped: source node '%s' not found",
                            op.source_id,
                        )
                        return
                if not self._graph.has_node(target_id):
                    resolved = self._resolve_node_ref(target_id)
                    if resolved:
                        target_id = resolved
                    else:
                        logger.warning(
                            "ADD_EDGE skipped: target node '%s' not found",
                            op.target_id,
                        )
                        return

                # Infer edge_type from node types if not provided
                edge_type = op.edge_type
                if edge_type is None:
                    edge_type = self._infer_edge_type(source_id, target_id)

                # Use Edge.create() for proper default weight handling
                edge = Edge.create(
                    source=source_id,
                    target=target_id,
                    edge_type=edge_type,
                    weight=op.weight,
                )
                self._graph.add_edge(edge)
                if self._graph_sync is not None:
                    self._graph_sync.on_edge_added(
                        source_id, target_id, edge.edge_type, edge.weight
                    )

        elif op.op == OperationType.REMOVE_EDGE:
            if op.source_id and op.target_id:
                self._graph.remove_edge(op.source_id, op.target_id, op.edge_type)
                if self._graph_sync is not None:
                    self._graph_sync.on_edge_removed(op.source_id, op.target_id, op.edge_type)

        elif op.op == OperationType.MERGE_NODES:
            if op.node_ids and len(op.node_ids) >= 2 and op.merged_data:
                self._execute_merge(op.node_ids, op.merged_data)

    def _resolve_add_node_id(self, op: Operation) -> str:
        """Resolve the node ID for an ADD_NODE operation.

        Checks (in order): op.node_id, common data keys, any data value
        that matches an edge-referenced ID, then falls back to a generated ID.
        """
        # 1. Explicit node_id on the operation
        if op.node_id:
            return op.node_id

        data = op.data or {}

        # 2. Well-known data keys
        for key in ("event_id", "id", "concept_id", "action_id", "intent_id", "time_id"):
            val = data.get(key)
            if val and isinstance(val, str):
                return val

        # 3. Dynamic key based on node_type (e.g., "intent" → "intent_id")
        if op.node_type:
            type_key = f"{op.node_type}_id"
            val = data.get(type_key)
            if val and isinstance(val, str):
                return val

        # 4. Scan all string values in data for an ID referenced by edges
        edge_ids = getattr(self, "_edge_referenced_ids", set())
        if edge_ids:
            for val in data.values():
                if isinstance(val, str) and val in edge_ids:
                    return val

        # 5. Generate a fallback ID
        from cognifold.query.config import UUID_HEX_LENGTH

        return f"{op.node_type}-{uuid.uuid4().hex[:UUID_HEX_LENGTH]}"

    def _resolve_node_ref(self, ref: str) -> str | None:
        """Resolve a node reference that may be a title/name instead of an ID.

        LLMs commonly generate ADD_EDGE operations using node titles instead
        of UUIDs. This method builds a title→ID cache and attempts exact
        and substring matching.

        Args:
            ref: The reference string (could be a title, name, or ID).

        Returns:
            The resolved node ID, or None if no match found.
        """
        ref_lower = ref.lower().strip()
        if not ref_lower:
            return None

        # Build/refresh title-to-ID cache
        if not hasattr(self, "_title_to_id_cache"):
            self._title_to_id_cache: dict[str, str] = {}
        cache = self._title_to_id_cache
        if not cache or len(cache) < self._graph.node_count:
            cache.clear()
            for node in self._graph.get_all_nodes():
                title = (node.data.get("title") or "").lower().strip()
                if title:
                    cache[title] = node.id
                name = (node.data.get("name") or "").lower().strip()
                if name and name != title:
                    cache[name] = node.id

        # Exact match
        if ref_lower in cache:
            logger.info("Resolved edge ref '%s' → node '%s'", ref, cache[ref_lower])
            return cache[ref_lower]

        # Substring match (ref contains title or title contains ref)
        for title_lower, node_id in cache.items():
            if title_lower in ref_lower or ref_lower in title_lower:
                logger.info(
                    "Resolved edge ref '%s' → node '%s' (substring match on '%s')",
                    ref,
                    node_id,
                    title_lower,
                )
                return node_id

        return None

    def _infer_edge_type(self, source_id: str, target_id: str) -> str:
        """Infer edge type from source and target node types.

        Uses heuristics based on common patterns:
        - Event → Concept: "grounds" (event provides evidence for concept)
        - Event → Intent: "triggers" (event activates an intent)
        - Concept → Intent: "triggers" (pattern suggests action)
        - Concept → Concept: "related_to" (generic relationship)
        - Time → Intent: "deadline_for" (temporal constraint)
        - Intent → Concept: "related_to" (intent relates to concept)

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            Inferred edge type string.
        """
        from cognifold.models.node import NodeType

        # Get node types (if nodes exist)
        source_type = None
        target_type = None

        try:
            source_node = self._graph.get_node(source_id)
            source_type = source_node.type
        except KeyError:
            pass

        try:
            target_node = self._graph.get_node(target_id)
            target_type = target_node.type
        except KeyError:
            pass

        # Infer based on type combinations
        if source_type == NodeType.EVENT:
            if target_type == NodeType.CONCEPT:
                return "grounds"
            elif target_type == NodeType.INTENT:
                return "triggers"
            elif target_type == NodeType.TIME:
                return "related_to"
        elif source_type == NodeType.CONCEPT:
            if target_type == NodeType.INTENT:
                return "triggers"
            elif target_type == NodeType.CONCEPT:
                return "related_to"
        elif source_type == NodeType.TIME:
            if target_type == NodeType.INTENT:
                return "deadline_for"
        elif source_type == NodeType.INTENT and target_type == NodeType.CONCEPT:
            return "related_to"

        # Default fallback
        return "related_to"

    def _execute_merge(self, node_ids: list[str], merged_data: dict[str, Any]) -> None:
        """Execute a MERGE_NODES operation."""
        from cognifold.models.node import Edge, Node, NodeType

        merged_id = node_ids[0]

        # Collect all edges
        incoming: set[str] = set()
        outgoing: set[str] = set()
        for node_id in node_ids:
            incoming.update(self._graph.get_predecessors(node_id))
            outgoing.update(self._graph.get_neighbors(node_id))

        # Remove old nodes (also removes their edges)
        for node_id in node_ids:
            self._graph.remove_node(node_id)

        # Generate embedding (skipped in fast-ingest mode)
        embedding: list[float] | None = None
        if not self._skip_embedding:
            from cognifold.utils.embeddings import get_embedding_service

            text_to_embed = (
                str(merged_data.get("title", "")) + " " + str(merged_data.get("description", ""))
            )
            embedding = get_embedding_service().embed_text(text_to_embed.strip())

        # Create merged node
        merged_node = Node(
            id=merged_id,
            type=NodeType.CONCEPT,
            data=merged_data,
            embedding=embedding,
        )
        self._graph.add_node(merged_node)

        # Reconnect edges (excluding self-loops and removed nodes)
        for source in incoming:
            if source not in node_ids and self._graph.has_node(source):
                self._graph.add_edge(Edge(source=source, target=merged_id))
        for target in outgoing:
            if target not in node_ids and self._graph.has_node(target):
                self._graph.add_edge(Edge(source=merged_id, target=target))

    def _detect_orphan_nodes(self, plan_id: str) -> None:
        """Detect and auto-fix concept/intent nodes with zero edges.

        Orphan higher-level nodes (concepts, intents) indicate that the LLM
        failed to create edges linking them to their grounding events. This
        method tries two strategies to reconnect them:

        1. Use ``grounded_in`` references on the node to create GROUNDS edges.
        2. Fall back to connecting to the most recent event node.

        Args:
            plan_id: ID of the plan that was executed.
        """
        from cognifold.models.node import Edge, NodeType

        orphan_ids: list[str] = []
        for node in self._graph.get_all_nodes():
            if node.type in (NodeType.CONCEPT, NodeType.INTENT):
                neighbors = list(self._graph.get_neighbors(node.id))
                predecessors = list(self._graph.get_predecessors(node.id))
                if not neighbors and not predecessors:
                    orphan_ids.append(node.id)

        if not orphan_ids:
            return

        fixed = 0
        for orphan_id in orphan_ids:
            node = self._graph.get_node(orphan_id)

            # Strategy 1: use grounded_in references
            for ref_id in node.grounded_in:
                if self._graph.has_node(ref_id):
                    try:
                        edge = Edge.create(
                            source=ref_id,
                            target=orphan_id,
                            edge_type="grounds",
                        )
                        self._graph.add_edge(edge)
                        fixed += 1
                        break
                    except (KeyError, ValueError):
                        continue

            # Strategy 2: connect to most recent event
            if not list(self._graph.get_neighbors(orphan_id)) and not list(
                self._graph.get_predecessors(orphan_id)
            ):
                events = sorted(
                    [n for n in self._graph.get_all_nodes() if n.type == NodeType.EVENT],
                    key=lambda n: n.created_at,
                    reverse=True,
                )
                if events:
                    try:
                        edge = Edge.create(
                            source=events[0].id,
                            target=orphan_id,
                            edge_type="grounds",
                        )
                        self._graph.add_edge(edge)
                        fixed += 1
                    except (KeyError, ValueError):
                        pass

        logger.warning(
            "Plan %s: %d orphan concept/intent node(s), %d auto-fixed with GROUNDS edges: %s",
            plan_id,
            len(orphan_ids),
            fixed,
            orphan_ids[:5],
        )

    def _validate_graph(self) -> ValidationReport:
        """Validate the graph integrity after execution.

        Returns:
            ValidationReport with any issues found.
        """
        from cognifold.graph.validator import GraphValidator

        validator = GraphValidator(self._graph)
        return validator.validate_all()

    def _log_validation_issues(self, report: ValidationReport, plan_id: str) -> None:
        """Log any integrity issues found during validation.

        Args:
            report: ValidationReport from validation.
            plan_id: ID of the plan that was executed.
        """
        if report.is_valid:
            logger.debug("Plan %s: Graph integrity check passed", plan_id)
            return

        # Log summary
        logger.warning(
            "Plan %s: Graph integrity issues found - %d errors, %d warnings",
            plan_id,
            report.error_count,
            report.warning_count,
        )

        # Log individual issues
        for issue in report.issues:
            msg = "[%s] %s"
            args: tuple[str, ...] = (issue.rule, issue.message)
            if issue.suggestion:
                msg += " - Suggestion: %s"
                args = (*args, issue.suggestion)
            if issue.level.value == "error":
                logger.error(msg, *args)
            else:
                logger.warning(msg, *args)
