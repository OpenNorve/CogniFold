"""Layered pipeline for fast ingest + progressive enrichment.

Layer 1: Add all events as nodes (no LLM, no embeddings, no PageRank)
Layer 2: Batched LLM enrichment (concepts, intents, edges)
Layer 3: Batch embeddings + FAISS index
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognifold.pipeline.progress import FastPipelineStats, LayerProgress, PrintProgressCallback

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.pipeline.progress import ProgressCallback
    from cognifold.replay.logger import GraphLogger
    from cognifold.simulator.timeline import Timeline

logger = logging.getLogger(__name__)


class LayeredPipeline:
    """Three-layer pipeline: fast ingest → LLM enrichment → embeddings.

    Usage::

        lp = LayeredPipeline(graph)
        lp.load_timeline("data/events.json")

        # Layer 1 only (~30s for 1200 events)
        stats = lp.run_layer1()

        # Or all layers
        stats = lp.run_all(use_agent=True)
    """

    def __init__(
        self,
        graph: ConceptGraph | None = None,
        agent_config: Any | None = None,
        prompt_profile: Any | None = None,
        progress: ProgressCallback | None = None,
        graph_logger: GraphLogger | None = None,
        batch_size: int = 10,
        graph_sync: Any | None = None,
    ) -> None:
        from cognifold.graph.store import ConceptGraph

        self._graph = graph or ConceptGraph()
        self._agent_config = agent_config
        self._prompt_profile = prompt_profile
        self._progress = progress or PrintProgressCallback()
        self._graph_logger = graph_logger
        self._batch_size = batch_size
        self._graph_sync = graph_sync
        self._timeline: Timeline | None = None
        self._stats = FastPipelineStats()

    def load_timeline(self, path: str | Path) -> int:
        """Load a timeline from a JSON file. Returns event count."""
        from cognifold.simulator.timeline import load_timeline

        self._timeline = load_timeline(path)
        logger.info("Loaded %d events from %s", len(self._timeline), path)
        return len(self._timeline)

    @property
    def graph(self) -> ConceptGraph:
        return self._graph

    @property
    def stats(self) -> FastPipelineStats:
        return self._stats

    @property
    def timeline(self) -> Timeline | None:
        return self._timeline

    @timeline.setter
    def timeline(self, value: Timeline | None) -> None:
        self._timeline = value

    # ------------------------------------------------------------------
    # Layer 1: Fast ingest — events as nodes, no LLM / embeddings / PR
    # ------------------------------------------------------------------

    def run_layer1(self, timeline: Timeline | None = None) -> FastPipelineStats:
        """Add all events as nodes. No LLM, no embeddings, no PageRank.

        Target: 1200 events in <30 seconds (pure Python, no I/O).
        """

        tl = timeline or self._timeline
        if tl is None:
            raise ValueError("No timeline loaded — call load_timeline() first")

        events = list(tl)
        progress = LayerProgress(layer=1, label="Fast ingest", total=len(events))
        self._progress.on_layer_start(progress)

        t0 = time.monotonic()

        for i, event in enumerate(events):
            self._add_event_node(event)

            progress.completed = i + 1
            progress.elapsed_ms = (time.monotonic() - t0) * 1000
            if (i + 1) % 100 == 0 or i + 1 == len(events):
                self._progress.on_layer_progress(progress)

        elapsed = (time.monotonic() - t0) * 1000
        progress.elapsed_ms = elapsed
        self._progress.on_layer_complete(progress)

        self._stats.layer1_events = len(events)
        self._stats.layer1_time_ms = elapsed
        self._stats.total_nodes = self._graph.node_count
        self._stats.total_edges = self._graph.edge_count

        logger.info(
            "Layer 1 complete: %d events → %d nodes in %.0fms",
            len(events),
            self._graph.node_count,
            elapsed,
        )
        return self._stats

    def _add_event_node(self, event: Event) -> None:
        """Add a single event as a node (no embedding, no edges)."""
        from cognifold.models.node import Node, NodeType

        node_id = event.event_id
        if self._graph.has_node(node_id):
            return  # idempotent

        data: dict[str, Any] = {
            "event_id": event.event_id,
            "title": event.title,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
        }
        if event.description:
            data["description"] = event.description
            data["source_text"] = event.description
        if event.location:
            data["location"] = event.location
        if event.duration_minutes is not None:
            data["duration_minutes"] = event.duration_minutes
        if event.context:
            data["context"] = event.context

        node = Node(
            id=node_id,
            type=NodeType.EVENT,
            data=data,
            created_at=event.timestamp,
            last_accessed=event.timestamp,
        )
        self._graph.add_node(node)

        # Notify graph sync writer if present
        if self._graph_sync is not None:
            self._graph_sync.on_node_added(node_id, "event", data)

        # Log to replay logger if present
        if self._graph_logger:
            self._graph_logger.log_operation(
                step=0,
                op_type="ADD_NODE",
                op_data={"node_id": node_id, "node_type": "event", "data": data},
                success=True,
            )

    # ------------------------------------------------------------------
    # Layer 2: Batched LLM enrichment
    # ------------------------------------------------------------------

    def run_layer2(self) -> FastPipelineStats:
        """Batched LLM enrichment — concepts, intents, and edges.

        Groups events into batches and sends each batch to the LLM in a
        single prompt, reducing API calls from N to N/batch_size.
        """
        from cognifold.agent.batch import BatchAgentProcessor
        from cognifold.executor.runner import PlanExecutor
        from cognifold.executor.validator import PlanValidator
        from cognifold.scoring.ranker import ContextRanker, ScoringConfig

        tl = self._timeline
        if tl is None:
            raise ValueError("No timeline loaded")

        events = list(tl)
        batches = [
            events[i : i + self._batch_size] for i in range(0, len(events), self._batch_size)
        ]

        progress = LayerProgress(layer=2, label="LLM enrichment", total=len(batches))
        self._progress.on_layer_start(progress)

        # Compute PageRank once (cached for all batches)
        ranker = ContextRanker(ScoringConfig())
        scored = ranker.score_nodes(self._graph)
        node_scores = {s.node_id: s.composite_score for s in scored}
        context_ids = [s.node_id for s in scored[:50]]

        batch_processor = BatchAgentProcessor(
            agent_config=self._agent_config,
            prompt_profile=self._prompt_profile,
        )
        executor = PlanExecutor(self._graph, skip_embedding=True, graph_sync=self._graph_sync)
        validator = PlanValidator(self._graph)

        t0 = time.monotonic()
        total_plans = 0

        for batch_idx, batch in enumerate(batches):
            try:
                plans = batch_processor.process_event_batch(
                    events=batch,
                    graph=self._graph,
                    context_node_ids=context_ids,
                    node_scores=node_scores,
                )

                for plan in plans:
                    validation = validator.validate(plan)
                    if validation.is_valid:
                        result = executor.execute(plan)
                        if result.success:
                            total_plans += 1
                        else:
                            self._stats.errors.append(
                                f"Batch {batch_idx} plan exec: {result.error}"
                            )
                    else:
                        logger.warning("Plan validation failed for batch %d", batch_idx)

                # Refresh scores periodically (every 5 batches)
                if (batch_idx + 1) % 5 == 0:
                    scored = ranker.score_nodes(self._graph)
                    node_scores = {s.node_id: s.composite_score for s in scored}
                    context_ids = [s.node_id for s in scored[:50]]

            except Exception as e:
                logger.error("Batch %d failed: %s", batch_idx, e)
                self._stats.errors.append(f"Batch {batch_idx}: {e}")

            progress.completed = batch_idx + 1
            progress.elapsed_ms = (time.monotonic() - t0) * 1000
            self._progress.on_layer_progress(progress)

        elapsed = (time.monotonic() - t0) * 1000
        progress.elapsed_ms = elapsed
        self._progress.on_layer_complete(progress)

        self._stats.layer2_batches = len(batches)
        self._stats.layer2_plans = total_plans
        self._stats.layer2_time_ms = elapsed
        self._stats.total_nodes = self._graph.node_count
        self._stats.total_edges = self._graph.edge_count

        logger.info(
            "Layer 2 complete: %d batches, %d plans applied, %d nodes, %d edges in %.0fms",
            len(batches),
            total_plans,
            self._graph.node_count,
            self._graph.edge_count,
            elapsed,
        )
        return self._stats

    # ------------------------------------------------------------------
    # Layer 3: Batch embeddings + FAISS index
    # ------------------------------------------------------------------

    def run_layer3(self) -> FastPipelineStats:
        """Batch embeddings for all nodes missing them, then build FAISS index."""
        from cognifold.embeddings.config import EmbeddingConfig
        from cognifold.embeddings.embedder import NodeEmbedder

        nodes = self._graph.get_all_nodes()
        nodes_missing = [n for n in nodes if n.embedding is None]

        progress = LayerProgress(layer=3, label="Batch embeddings", total=len(nodes_missing))
        self._progress.on_layer_start(progress)

        t0 = time.monotonic()

        if nodes_missing:
            embed_cfg = getattr(self, "_embedding_config", None) or EmbeddingConfig()
            embedder = NodeEmbedder(embed_cfg)
            embeddings = embedder.embed_nodes(nodes_missing)

            # Write embeddings back to graph node attributes
            nx_graph = self._graph._graph  # type: ignore[reportPrivateUsage]
            for node_id, emb in embeddings.items():
                if node_id in nx_graph.nodes:
                    nx_graph.nodes[node_id]["embedding"] = emb.tolist()

            progress.completed = len(embeddings)
        else:
            progress.completed = 0

        elapsed = (time.monotonic() - t0) * 1000
        progress.elapsed_ms = elapsed
        self._progress.on_layer_complete(progress)

        # Build FAISS index if available
        try:
            from cognifold.embeddings.search import SemanticSearch

            embed_cfg = getattr(self, "_embedding_config", None) or EmbeddingConfig()
            embedder = NodeEmbedder(embed_cfg)
            search = SemanticSearch(embedder)
            search.build_index(self._graph)
            logger.info("FAISS index built with %d nodes", self._graph.node_count)
        except ImportError:
            logger.debug("FAISS not available, skipping index build")

        self._stats.layer3_nodes_embedded = progress.completed
        self._stats.layer3_time_ms = elapsed
        self._stats.total_nodes = self._graph.node_count
        self._stats.total_edges = self._graph.edge_count

        logger.info(
            "Layer 3 complete: %d nodes embedded in %.0fms",
            progress.completed,
            elapsed,
        )
        return self._stats

    # ------------------------------------------------------------------
    # Convenience: run all layers
    # ------------------------------------------------------------------

    def run_all(self, use_agent: bool = False) -> FastPipelineStats:
        """Run all applicable layers sequentially.

        Args:
            use_agent: If True, run Layer 2 (LLM enrichment).
                       If False, only Layer 1 + Layer 3.
        """
        self.run_layer1()

        if use_agent:
            self.run_layer2()

        self.run_layer3()

        return self._stats
