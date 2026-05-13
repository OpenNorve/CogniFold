# Cognifold Architecture

Comprehensive system architecture and critical implementation details.

---

## System Overview

Cognifold is a dynamic concept graph system that processes event streams and maintains an evolving knowledge representation. It uses LLM agents to analyze events, identify patterns, and build a graph of concepts, intents, and temporal relationships.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      COGNIFOLD SYSTEM                                         │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════
                                    EVENT PROCESSING (Write Path)
═══════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────┐     ┌──────────────┐
    │  Generator   │     │   Importer   │
    │   (LLM)      │     │   (Data)     │
    └──────┬───────┘     └──────┬───────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼
           ┌──────────────┐
           │ Event Stream │  (unified Event schema)
           └──────┬───────┘
                  │
                  ▼
    ┌─────────────────────┐
    │  Context Selector   │
    │  (ContextRanker)    │
    │                     │      ┌─────────────────────────────────────────────┐
    │  • PageRank         │      │              CONCEPT GRAPH                  │
    │  • Recency decay    │      │  ┌───────────────────────────────────────┐  │
    │  • Access frequency │◄────►│  │  Nodes (Pydantic models)              │  │
    │  • Urgency boost    │      │  │  ├── event    ├── concept             │  │
    │                     │      │  │  ├── intent   └── time                │  │
    │  Output: Context    │      │  ├───────────────────────────────────────┤  │
    │  Window (top-k)     │      │  │  Edges (typed, weighted)              │  │
    └──────────┬──────────┘      │  │  • 8 semantic types                   │  │
               │                 │  │  • Weight: 0.0-1.0                    │  │
               ▼                 │  └───────────────────────────────────────┘  │
    ┌─────────────────────┐      │                                             │
    │  LangGraph Agent    │      │  Storage: NetworkX MultiDiGraph             │
    │  (CognifoldAgent)   │      │  Persistence: JSON                          │
    │                     │      └──────────────────▲──────────────────────────┘
    │  Input:             │                         │
    │  • Event            │                         │
    │  • Hierarchical     │                         │
    │    context window   │                         │
    │                     │                         │
    │  Output:            │                         │
    │  • UpdatePlan       │                         │
    └──────────┬──────────┘                         │
               │                                    │
               ▼                                    │
    ┌─────────────────────┐                         │
    │     Executor        │─────────────────────────┘
    │   (PlanExecutor)    │
    │                     │
    │  • Validates plan   │
    │  • Atomic execution │
    │  • Rollback on fail │
    └─────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════════
                                   QUERY & RETRIEVAL (Read Path)
═══════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────┐
    │  Natural Language │
    │      Query        │
    └────────┬─────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         MemoryQueryAgent                                     │
    │  ┌─────────────────────────────────────────────────────────────────────┐    │
    │  │                    Entry Point Selection                             │    │
    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
    │  │  │   LEGACY    │  │    BM25     │  │  SEMANTIC   │  │   HYBRID    │ │    │
    │  │  │  (keyword)  │  │  (inverted  │  │ (embedding  │  │ (BM25 +     │ │    │
    │  │  │             │  │   index)    │  │  cosine)    │  │  RRF fusion)│ │    │
    │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
    │  │                                                    ┌─────────────┐   │    │
    │  │                                                    │  AGENTIC    │   │    │
    │  │                                                    │ (2-round +  │   │    │
    │  │                                                    │  sufficiency│   │    │
    │  │                                                    │  check)     │   │    │
    │  │                                                    └─────────────┘   │    │
    │  └─────────────────────────────────────────────────────────────────────┘    │
    │                                    │                                         │
    │                                    ▼                                         │
    │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
    │  │  GraphTraverser  │───►│   QueryScorer    │───►│ ContextAssembler │       │
    │  │  (BFS + decay)   │    │  (relevance)     │    │  (format result) │       │
    │  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
    └─────────────────────────────────────────────────────────────────────────────┘
             │                                                    ▲
             │            ┌────────────────────┐                  │
             └───────────►│   CONCEPT GRAPH    │◄─────────────────┘
                          └────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │   QueryResult    │
    │  • summary       │
    │  • nodes         │
    │  • relationships │
    └──────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════════
                                  REPLAY & VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════════════════════

    ┌──────────────────┐
    │  Run Logs        │
    │  (JSONL)         │
    │  • events        │
    │  • plans         │
    │  • snapshots     │
    └────────┬─────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           Replay System                                      │
    │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
    │  │   ReplayLogger   │───►│  ReplayParser    │───►│  ReplayRenderer  │       │
    │  │  (capture logs)  │    │  (parse states)  │    │  (pyvis HTML)    │       │
    │  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
    └─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Interactive     │
    │  HTML Replay     │
    │  • play/pause    │
    │  • step controls │
    │  • node colors   │
    │  • context glow  │
    └──────────────────┘
```

---

## Core Data Models

All models are Pydantic `BaseModel` subclasses in `src/cognifold/models/`.

### Node (`models/node.py`)

```python
class Node(BaseModel):
    id: str                           # Unique identifier (e.g., "e-001", "c-001")
    type: NodeType                    # EVENT, CONCEPT, INTENT, TIME
    data: dict[str, Any]              # Payload (title, description, etc.)
    created_at: datetime              # Creation timestamp
    last_accessed: datetime           # For recency scoring
    access_count: int                 # For access frequency scoring

    # Explainability (required for non-event nodes)
    reasoning: str | None             # Why this node was created
    grounded_in: list[str]            # Event IDs that justify this node
    update_history: list[UpdateHistoryEntry]  # Audit trail
```

**Node Types** (`NodeType` enum):
| Type | Description | ID Prefix |
|------|-------------|-----------|
| `EVENT` | Direct representation of input events | `e-` |
| `CONCEPT` | Patterns that emerge from multiple events | `c-` |
| `INTENT` | Goals/desires (formerly "action") | `i-` |
| `TIME` | Temporal anchors (deadlines, schedules) | `t-` |

**Intent Status Lifecycle** (`IntentStatus` enum):
```
PENDING → ACTION_SCHEDULED → RESOLVED
       ↘ REJECTED
       ↘ DEFERRED
```

### Edge (`models/node.py`)

```python
class Edge(BaseModel):
    source: str                       # Source node ID
    target: str                       # Target node ID
    edge_type: str | None             # Semantic type (None for legacy)
    weight: float                     # 0.0-1.0, defaults by type
    created_at: datetime              # For edge recency in PageRank
    metadata: dict[str, Any]          # Optional additional data
```

**Edge Types** (`BaseEdgeType` enum with default weights):

| Type | Weight | Typical Usage |
|------|--------|---------------|
| `GROUNDS` | 0.9 | Event → Concept/Intent (direct evidence) |
| `CAUSES` | 0.9 | Event → Event (causal link) |
| `TRIGGERS` | 0.8 | Concept/Event → Intent |
| `REINFORCES` | 0.7 | Event → Concept (supporting evidence) |
| `PART_OF` | 0.7 | Event/Concept → Concept (structural, subtopic) |
| `DERIVED_FROM` | 0.6 | Concept → Concept |
| `DEADLINE_FOR` | 0.6 | Time → Intent |
| `RELATED_TO` | 0.5 | Generic (default) |
| `USER_FEEDBACK` | 0.8 | Feedback event → Intent |

**Multiple Edges**: The graph supports multiple edges between the same node pair with different types, keyed by `(source, target, edge_type)`.

### Event (`models/event.py`)

```python
class Event(BaseModel):
    event_id: str                     # Unique identifier
    timestamp: datetime               # When event occurred
    source: str                       # Domain identifier (e.g., "personal-timeline")
    event_type: str                   # Free-form type (e.g., "meal", "http.request")
    title: str                        # Short description
    description: str | None           # Detailed description
    location: str | None              # Physical location
    duration_minutes: int | None      # Duration
    context: dict[str, Any]           # Domain-specific structured data
    metadata: dict[str, Any]          # Additional unstructured data
```

**Domain-Agnostic Design**: The `context` field holds domain-specific data that the LLM interprets. Same Event model works for personal timelines, computer activity, service logs, wiki chunks, etc.

### UpdatePlan (`models/plan.py`)

```python
class UpdatePlan(BaseModel):
    plan_id: str                      # Unique identifier
    trigger_event_id: str             # Event that triggered this plan
    reasoning: str                    # Agent's explanation
    operations: list[Operation]       # Operations to execute atomically
```

**Operation Types** (`OperationType` enum):

| Operation | Required Fields | Description |
|-----------|-----------------|-------------|
| `ADD_NODE` | `node_type`, `data`, `reasoning`*, `grounded_in`* | Create node |
| `UPDATE_NODE` | `node_id`, `data`, `update_reasoning`* | Modify node |
| `REMOVE_NODE` | `node_id` | Delete node and edges |
| `ADD_EDGE` | `source_id`, `target_id`, `edge_type`?, `weight`? | Connect nodes |
| `REMOVE_EDGE` | `source_id`, `target_id`, `edge_type`? | Disconnect nodes |
| `MERGE_NODES` | `node_ids[]`, `merged_data`, `reasoning` | Combine similar nodes |

*Required for non-event nodes

---

## Graph Storage (`graph/store.py`)

The `ConceptGraph` class wraps NetworkX `MultiDiGraph` for in-memory graph operations.

**Key Design Decisions**:
- **MultiDiGraph**: Supports multiple edges between same node pair (different types)
- **Node attributes stored directly**: Type, data, timestamps, reasoning, grounded_in
- **Edge keying**: `(source, target, edge_type)` allows typed edge lookups
- **Legacy compatibility**: `edge_type=None` uses key `"__legacy__"`

**Critical Methods**:
```python
class ConceptGraph:
    def add_node(self, node: Node) -> None
    def get_node(self, node_id: str) -> Node
    def update_node(self, node_id: str, data: dict) -> None
    def remove_node(self, node_id: str) -> None
    def add_edge(self, edge: Edge) -> None
    def get_edge(self, source: str, target: str, edge_type: str | None) -> Edge
    def remove_edge(self, source: str, target: str, edge_type: str | None) -> None
    def get_neighbors(self, node_id: str) -> list[str]
    def get_predecessors(self, node_id: str) -> list[str]
    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]
```

**Persistence** (`graph/persistence.py`):
- `save_graph(graph, path)`: Serialize to JSON with ISO timestamps
- `load_graph(path) -> ConceptGraph`: Deserialize from JSON
- `graph_to_dict(graph) -> dict`: Convert graph to serializable dict (version, saved_at, nodes, edges). Used by the `/graph/export` HTTP endpoint and session persistence.

**Edge Inference** (`graph/edge_inference.py`):

Post-ingestion edge inference for disconnected graphs. When LLM-generated UpdatePlans fail to create edges (e.g., referencing titles instead of IDs), the graph ends up with few or no edges.

- `EdgeInferenceEngine`: Creates `RELATED_TO` edges between nodes with high embedding similarity
- Uses kNN approach: connects each node to top-k nearest neighbors by cosine similarity
- Parameters: `similarity_threshold` (default 0.30), `max_edges_per_node` (default 3)
- Requires stored node embeddings (`Node.embedding` field)
- Called from `ConceptGraph.infer_edges()` and `MemoryQueryAgent._expand_with_neighbors()`

**Graph Projection** (`graph/projection.py`):

Read-only abstraction layer for graph access. Any downstream system (neural network, symbolic tracker, consolidation engine) reads through this interface.

- `GraphProjection`: `@runtime_checkable` Protocol with 5 methods (`get_active_concepts`, `get_concept_connections`, `get_cluster_summary`, `get_lifecycle_state`, `get_node_count_by_type`)
- `NetworkXProjection`: Default implementation wrapping `ConceptGraph` + optional `ContextRanker`
- `GraphSnapshot`: Serializable point-in-time graph state (node/edge counts, top concepts, active intents, cluster count) with `to_dict()`/`from_dict()` round-trip
- `graph_to_snapshot()`: Factory function to create a snapshot from a `ConceptGraph`

**Observability** (`utils/llm_metrics.py`, `utils/budget.py`):

- `LLMMetricsCollector`: Thread-safe collector for LLM call metrics (model, tokens, latency, cost). Cost estimation covers 15+ models (OpenAI, Gemini, Claude).
- `BudgetEnforcer`: Per-session token/cost/call limit enforcement. Raises `BudgetExceededError` when limits breached.

---

## Scoring & Context Window (`scoring/`)

### ContextRanker (`scoring/ranker.py`)

Computes node relevance scores using composite formula:

```
Score(node) = α × StructuralRank + β × RecencyScore + γ × AccessScore
```

**Default Weights** (`ScoringConfig`):
| Weight | Default | Description |
|--------|---------|-------------|
| `alpha` | 0.4 | Structural importance (PageRank) |
| `beta` | 0.4 | Recency (time decay) |
| `gamma` | 0.2 | Access frequency |

**Score Components**:

1. **StructuralRank**: Weighted PageRank on graph topology
   - Uses edge weights: `effective_weight = edge.weight × recency_factor(edge.created_at)`
   - Higher weight edges contribute more to node importance

2. **RecencyScore**: `exp(-decay_rate × hours_since_last_accessed)`
   - Default `decay_rate = 0.01`

3. **AccessScore**: `access_count / max_access_count` (normalized)

4. **UrgencyScore** (multiplier): Boost for nodes connected to approaching TIME nodes
   - `urgency_boost = 2.0` max multiplier
   - `urgency_window_hours = 24.0` hours before deadline

### Hierarchical Context Windows (`scoring/hierarchical.py`)

Three-level context with different scoring strategies:

| Level | % of Budget | Content | Scoring Strategy |
|-------|-------------|---------|------------------|
| Immediate | 10% | Recent events, urgent intents | 70% recency, 30% urgency |
| Working | 30% | Active concepts, patterns | 50% PageRank, 30% recency, 20% type |
| Background | 50% | Historical context | 80% PageRank, 20% diversity |

**Features**:
- Deduplication: Node appears in highest priority level only
- Each level includes Nodes + Relationships sections
- Edges show type, weight, and connected node summaries

---

## Pipeline & Execution

### Pipeline (`pipeline.py`)

Orchestrates end-to-end event processing:

```python
class Pipeline:
    def __init__(self, config: CognifoldConfig)
    def load_timeline(self, path: str) -> int
    def process_event(self, event: Event) -> PipelineResult
    def step(self) -> PipelineResult | None
    def run(self, max_events: int | None) -> PipelineStats
    def save_graph(self, path: str) -> None
    def visualize(self, path: str) -> Path
```

**Processing Flow for Each Event**:
1. Get context window from `ContextRanker`
2. Generate `UpdatePlan` via agent (or default plan if no API key)
3. Validate plan with `PlanValidator`
4. Execute plan atomically with `PlanExecutor`
5. Update statistics

### PlanExecutor (`executor/runner.py`)

**Atomic Execution with Rollback**:
1. Take graph snapshot
2. Sort operations by type priority (ADD_NODE → UPDATE_NODE → ADD_EDGE → REMOVE_EDGE → REMOVE_NODE → MERGE_NODES) to handle LLM-generated plans where edges may appear before their target nodes
3. Execute operations in sorted order
4. On failure: restore snapshot and abort
5. Optionally validate graph integrity after execution

**Node ID Resolution** (`_resolve_add_node_id`):
For ADD_NODE operations, the executor resolves the node ID through a multi-layer lookup:
1. `op.node_id` — explicit ID on the operation
2. Well-known data keys: `event_id`, `id`, `concept_id`, `action_id`, `intent_id`, `time_id`
3. Dynamic `{node_type}_id` key (e.g., intent → `intent_id`)
4. Scan all data values for IDs referenced by edge operations in the plan
5. Auto-generate fallback: `{node_type}-{uuid[:8]}`

```python
class PlanExecutor:
    def execute(self, plan: UpdatePlan) -> ExecutionResult
```

### PlanValidator (`executor/validator.py`)

Validates UpdatePlan before execution:
- Node existence for UPDATE/REMOVE operations
- Edge endpoint existence for ADD_EDGE
- Duplicate node detection
- Grounding validation (concepts/intents must link to events)
- Reasoning validation (non-events need reasoning field)

Returns `ValidationResult` with errors (blocking) and warnings (soft).

---

## Agent System (`agent/`)

### CognifoldAgent (`agent/agent.py`)

LangGraph-based agent that generates UpdatePlans.

**Input**:
- Event being processed
- Hierarchical context window (immediate/working/background)
- Graph tools for exploration

**Output**: `UpdatePlan` with operations

**Tools Available** (`agent/tools.py`):
| Tool | Description |
|------|-------------|
| `get_node(node_id)` | Get full node details |
| `get_neighbors(node_id, direction)` | Find connected nodes |
| `find_nodes_by_type(node_type)` | List nodes of a type |
| `search_nodes(keyword)` | Keyword search in titles/data |
| `get_graph_stats()` | Graph statistics |

### Prompts (`agent/prompts.py`)

System prompt includes:
- Role definition
- Node type descriptions with examples
- Edge type semantics
- Guidelines for concept creation, strength management, hierarchy
- Reasoning mode instructions (quick, analytical, consolidation)
- Plan self-review section for retroactive connections
- Concept refinement for overloaded nodes (5+ connections)

### Domain Configurations (`agent/domain.py`)

Domain-specific prompt customization without changing core logic:

| Domain | Source |
|--------|--------|
| `personal-timeline` | Event generator |
| `computer-activity` | Computer activity generator |
| `service-logs` | Service logs generator |
| `wiki` | Wiki importer |

---

## Query System (`query/`)

### MemoryQueryAgent (`query/agent.py`)

Retrieves relevant context for natural language queries.

**Retrieval Modes** (`RetrievalMode` enum):

| Mode | Backend | Requirements |
|------|---------|--------------|
| `LEGACY` | Keyword matching | None |
| `BM25` | Inverted index + BM25 | None |
| `SEMANTIC` | Embedding similarity | `NodeEmbedder` |
| `HYBRID` | BM25 + semantic with RRF (default) | `NodeEmbedder` (auto-degrades to BM25) |

Additionally, the `retrieval/` module has a `RetrievalStrategy` enum with an `AGENTIC` mode:

| Strategy | Backend | Requirements |
|----------|---------|--------------|
| `AGENTIC` | 2-round hybrid with LLM sufficiency check + complementary queries + multi-RRF | `HybridRetriever` + LLM API key |

The agentic retriever (`retrieval/agentic.py`) orchestrates: Round 1 hybrid retrieval → LLM sufficiency check → if insufficient, generate complementary queries → Round 2 parallel retrieval → multi-RRF fusion.

**Query Flow**:
1. **Entry Point Selection** (`EntryPointSelector`): Find starting nodes using configured retrieval mode
2. **Graph Traversal** (`GraphTraverser`): BFS from entry points with score decay
3. **Scoring** (`QueryScorer`): Rank traversed nodes by relevance
4. **Assembly** (`ContextAssembler`): Format results into `QueryResult`

```python
agent = MemoryQueryAgent(graph, config, embedder=embedder)
result = agent.query("exercise habits")
# result.summary, result.nodes, result.statistics
```

---

## Retrieval System (`retrieval/`, `embeddings/`)

### BM25 Index (`retrieval/bm25.py`)

Standard BM25 scoring with inverted index:
- `k1=1.5`, `b=0.75` (default parameters)
- Document = node text (title + description + reasoning)
- Tokenization with stopword removal

### Embedding System (`embeddings/`)

**Provider Abstraction** (`EmbeddingProvider`):
- `MockEmbeddingProvider`: Deterministic embeddings for testing (numpy RNG)
- `GeminiEmbeddingProvider`: Google AI embeddings
- `OpenAIEmbeddingProvider`: OpenAI embeddings

**NodeEmbedder** (`embeddings/embedder.py`):
- Generates embeddings for nodes (title + description + reasoning)
- LRU caching for efficiency
- Batch embedding support
- Export/import for persistence

**SemanticSearch** (`embeddings/search.py`):
- Cosine similarity over node embeddings
- Optional FAISS ANN index (`IndexFlatIP` + `IndexIDMap`) for fast lookup, numpy fallback
- Index building for fast lookup
- Type filtering in results

### Hybrid Retrieval (`retrieval/hybrid.py`)

**Reciprocal Rank Fusion (RRF)**:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```
Where `k=60` (default RRF constant).

Combines BM25 and semantic rankings for best results.

---

## Event Sources

### Generators (`generator/`)

LLM-based synthetic event generation:

| Generator | Input | Output |
|-----------|-------|--------|
| `EventGenerator` | Persona | Personal timeline events |
| `ComputerActivityGenerator` | WorkProfile | Computer usage events |
| `ServiceLogsGenerator` | ServiceTopology | Microservice events |

### Importers (`importers/`)

Data transformation without LLM:

| Importer | Input | Output |
|----------|-------|--------|
| `WikiImporter` | Markdown/PDF files | Chunked wiki events |

**Wiki Chunking Strategies**:
- `heading`: Split at markdown headings
- `paragraph`: Split at paragraph boundaries
- `fixed`: Fixed-size chunks with overlap

---

## Intent Execution System (`intent/`)

Transforms graph intents into executable actions.

**Components**:
- `IntentToActionAgent`: Converts intents to concrete `Action` objects
- `ActionQueue`: Manages scheduled actions sorted by time
- `ActionExecutor`: Simulates action execution, generates result events
- `IntentSelector`: Identifies actionable intents based on urgency

**Action vs Intent**:
| Aspect | Intent | Action |
|--------|--------|--------|
| Storage | Graph node | ActionQueue |
| Timing | No specific time | Scheduled execution |
| Nature | Goal/desire | Concrete step |

---

## Replay & Visualization

### Replay (`replay/`)

Graph evolution visualization from run logs.

**ReplayLogger** (`replay/logger.py`):
- Logs events, plans, graph snapshots to JSONL
- Each entry timestamped for ordering

**ReplayRenderer** (`replay/renderer.py`):
- Generates interactive HTML with pyvis
- Play/pause/step controls
- Node colors by type, sizes by score

### CLI (`cli/`)

```bash
cognifold run timeline.json --agent --steps 15
cognifold query "exercise habits" --graph graph.json
cognifold build-timeline --source wiki --input data/wiki/ -o timeline.json
cognifold replay logs/replay.jsonl
```

---

## Configuration (`config.py`)

```yaml
# cognifold.yaml
model:
  name: "gemini-2.0-flash"
  temperature: 0.7
  max_tokens: 4096
  max_exploration_steps: 3

scoring:
  alpha: 0.4
  beta: 0.4
  gamma: 0.2
  decay_rate: 0.01

context:
  max_nodes: 50
  min_score_threshold: 0.01

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## HTTP Service Layer (`service/`)

The service layer exposes Cognifold as a stateful REST API, allowing clients to create sessions, ingest events, query the graph, and inspect graph state over HTTP.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HTTP Service Layer                                 │
│                                                                              │
│   ┌────────────┐    ┌────────────────────────────────────────────────┐      │
│   │  FastAPI    │    │           Route Handlers                       │      │
│   │  App        │    │  ┌──────────┐ ┌────────┐ ┌───────┐ ┌───────┐ │      │
│   │  (app.py)   │───►│  │ sessions │ │ events │ │ query │ │ graph │ │      │
│   │             │    │  └──────────┘ └────────┘ └───────┘ └───────┘ │      │
│   │  Middleware: │    └──────────────────┬───────────────────────────┘      │
│   │  • Auth     │                       │                                   │
│   │  • Logging  │                       ▼                                   │
│   └────────────┘    ┌────────────────────────────────────────────────┐      │
│                      │         Session Manager                        │      │
│                      │  ┌──────────────────────────────────────────┐ │      │
│                      │  │ Session 1    Session 2    Session N      │ │      │
│                      │  │ ┌─────────┐ ┌─────────┐                 │ │      │
│                      │  │ │ Graph   │ │ Graph   │  ...            │ │      │
│                      │  │ │ Ranker  │ │ Ranker  │                 │ │      │
│                      │  │ │ Agent   │ │ Agent   │                 │ │      │
│                      │  │ └─────────┘ └─────────┘                 │ │      │
│                      │  └──────────────────────────────────────────┘ │      │
│                      │  • TTL eviction  • Persist on delete          │      │
│                      │  • LRU when full • Per-session locks          │      │
│                      └──────────────────────────────────────────────┘      │
│                                         │                                   │
│                      ┌──────────────────┴──────────────────┐               │
│                      │        Event Processor               │               │
│                      │  1. Build Event from input            │               │
│                      │  2. Compute context window (Ranker)   │               │
│                      │  3. Generate plan (Agent or default)  │               │
│                      │  4. Validate plan                     │               │
│                      │  5. Execute atomically                │               │
│                      └──────────────────────────────────────┘               │
│                                         │                                   │
│                      ┌──────────────────┴──────────┐                       │
│                      │    Task Tracker (async)      │                       │
│                      │  • Pending/Running/Done      │                       │
│                      │  • Poll for status            │                       │
│                      └─────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

**AppSettings** (`service/app.py`):
```python
@dataclass
class AppSettings:
    persist_dir: str = "./sessions"      # Graph persistence directory
    max_sessions: int = 100              # Max concurrent sessions
    session_ttl_hours: float = 24.0      # Inactive session expiry
    api_keys: set[str] | None = None     # None = auth disabled
```

**Session** (`service/session.py`):
Each session owns an isolated graph, ranker, and lazy-loaded agent:
- `graph: ConceptGraph` — the session's concept graph
- `ranker: ContextRanker` — scoring with session-specific config
- `agent: CognifoldAgent | None` — lazy-initialized on first event with LLM keys
- `lock: asyncio.Lock` — per-session concurrency control
- `llm_env()` — context manager that temporarily sets API key env vars

**SessionManager** (`service/session.py`):
- Creates sessions with unique 16-char hex IDs
- LRU eviction when `max_sessions` reached (oldest by `last_accessed`)
- TTL cleanup for sessions inactive beyond `session_ttl_hours`
- Persists graph to `{persist_dir}/{session_id}/graph.json` on delete/evict
- `persist_all()` called on shutdown

**Event Processor** (`service/processor.py`):
Wraps the Pipeline's event processing for the service layer:
1. Convert `EventInput` → internal `Event` model (generates `event_id`)
2. Compute context window via `ContextRanker`
3. Generate plan inside `session.llm_env()`: use agent if LLM keys provided, else default plan (adds event as node)
4. Validate plan with `PlanValidator` (falls back to default plan if invalid)
5. Execute atomically with `PlanExecutor` inside `session.llm_env()` (embeddings need API keys)
6. Log and return execution errors (propagated to client via `error` field)

**Authentication** (`service/auth.py`):
- `X-API-Key` header, validated by `APIKeyValidator` dependency
- If `api_keys=None`: auth disabled (dev mode), all requests pass
- Health endpoints (`/health`, `/ready`) always bypass auth

**Task Tracker** (`service/tasks.py`):
- In-memory tracking for async event ingestion
- States: `pending` → `running` → `completed` | `failed`
- Client polls `/tasks/{task_id}` for result

### API Endpoints

All API routes are prefixed with `/api/v1`.

**Health** (no auth):

| Method | Path | Response |
|--------|------|----------|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/ready` | `{"status": "ok", "active_sessions": N}` |

**Sessions**:

| Method | Path | Request Body | Response | Status |
|--------|------|-------------|----------|--------|
| POST | `/sessions` | `CreateSessionRequest` | `SessionInfo` | 201 |
| GET | `/sessions/{id}` | — | `SessionInfo` | 200 |
| DELETE | `/sessions/{id}` | — | — | 204 |
| POST | `/sessions/{id}/load` | `LoadGraphRequest` | `SessionInfo` | 200 |

**Events**:

| Method | Path | Request Body | Response | Status |
|--------|------|-------------|----------|--------|
| POST | `/sessions/{id}/events` | `IngestEventRequest` | `IngestEventResponse` or `AsyncTaskResponse` | 200 |
| POST | `/sessions/{id}/events/batch` | `BatchIngestRequest` | `BatchIngestResponse` | 200 |
| GET | `/sessions/{id}/tasks/{task_id}` | — | `TaskStatusResponse` | 200 |

**Query**:

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| POST | `/sessions/{id}/query` | `QueryRequest` | `QueryResponse` |

**Graph State**:

| Method | Path | Query Params | Response |
|--------|------|-------------|----------|
| GET | `/sessions/{id}/graph` | `max_nodes` (1-1000) | `GraphStateResponse` |
| GET | `/sessions/{id}/graph/stats` | — | `GraphStatsResponse` |
| GET | `/sessions/{id}/graph/concepts` | `top` (1-100) | `list[QueryNodeResponse]` |
| GET | `/sessions/{id}/graph/intents` | `recent` (1-100) | `list[QueryNodeResponse]` |
| GET | `/sessions/{id}/graph/events` | `recent` (1-100) | `list[QueryNodeResponse]` |
| GET | `/sessions/{id}/graph/nodes/{node_id}` | — | `NodeResponse` |
| GET | `/sessions/{id}/graph/nodes/{node_id}/expand` | `layers`, `direction`, `max_nodes` | `NodeExpansionResponse` |
| GET | `/sessions/{id}/graph/export` | — | `dict` (persistence format) |

**Domains** (in-memory only, re-register on server restart):

| Method | Path | Request Body | Response | Status |
|--------|------|-------------|----------|--------|
| POST | `/domains` | `DomainRegisterRequest` | `{"status", "name"}` | 201 |
| GET | `/domains` | — | `{"domains": [...]}` | 200 |
| GET | `/domains/{name}` | — | `{"name", "description"}` | 200 |

### Key Request/Response Models (`service/models.py`)

**SessionConfig**: `model_name`, `temperature`, `max_nodes`, `domain`, `scoring_alpha/beta/gamma`

**EventInput**: `event_type` (required), `title` (required), `timestamp`, `source`, `description`, `location`, `duration_minutes`, `context`, `metadata`

**IngestEventResponse**: `event_id`, `plan_id`, `reasoning`, `operations_completed`, `success`, `execution_time_ms`, `graph_stats`, `error` (execution failure message, if any)

**QueryRequest**: `query` (required), `max_nodes`, `max_context_chars`, `query_mode`

**QueryResponse**: `context`, `nodes[]`, `traversal_path[]`, `query_metadata`, `query_time_ms`

**GraphStatsResponse**: `node_count`, `edge_count`, `concepts`, `events`, `intents`, `time_nodes`

**NodeResponse**: `node_id`, `node_type`, `data`, `created_at`, `last_accessed`, `access_count`, `reasoning`, `grounded_in[]`, `neighbors[]`, `predecessors[]`

### Interactive CLI Client (`cli/client.py`)

`CognifoldClient` wraps all API endpoints using stdlib `urllib.request` (no extra dependencies). `ClientREPL` provides an interactive REPL with `:` command dispatch for session management, event ingestion, querying, and graph exploration. Natural language queries (input without `:` prefix) are sent directly to the query endpoint.

---

## Module Dependencies

```
models/          # No internal dependencies
    └── node.py, event.py, plan.py

graph/           # Depends on: models, (numpy for edge_inference)
    └── store.py, persistence.py, validator.py, metrics.py, edge_inference.py

scoring/         # Depends on: graph, models
    └── ranker.py, hierarchical.py

executor/        # Depends on: graph, models
    └── runner.py, validator.py

agent/           # Depends on: graph, models, scoring, executor
    └── agent.py, prompts.py, prompt_sections.py, tools.py, domain.py, prompt_profile.py

query/           # Depends on: graph, models, scoring, embeddings, retrieval
    └── agent.py, strategies.py, scoring.py, assembly.py, probe.py, llm.py, prompts.py

embeddings/      # Depends on: graph, models
    └── providers.py, embedder.py, search.py, config.py

retrieval/       # Depends on: graph, models, embeddings
    └── bm25.py, hybrid.py, agentic.py, config.py, result.py

temporal/        # Depends on: (standalone, uses dateparser)
    └── extractor.py

intent/          # Depends on: graph, models, agent
    └── agent.py, queue.py, selector.py, executor.py, models.py, prompts.py

generator/       # Depends on: models
    └── base.py, event_generator.py, computer_activity.py, service_logs.py

importers/       # Depends on: models
    └── base.py, wiki.py

replay/          # Depends on: graph, models
    └── logger.py, player.py, renderer.py

utils/           # Shared utilities
    └── embeddings.py

pipeline.py      # Depends on: all above modules

service/         # Depends on: graph, models, scoring, executor, agent, query
    └── app.py, session.py, processor.py, auth.py, tasks.py, models.py, routes/

simulator/       # Depends on: graph, models, scoring, executor, agent, replay
    └── cli.py, visualizer.py, timeline.py

cli/             # Depends on: all modules (CLI entry points)
    └── __init__.py, run.py, query.py, generate.py, replay.py, serve.py,
        client.py, build.py, config.py
```

---

## Architecture Decision Records

### ADR-001: Edge Inference is Opt-In Only

**Status**: Accepted

**Context**: `EdgeInferenceEngine` (`graph/edge_inference.py`) uses kNN cosine similarity to infer `RELATED_TO` edges between nodes. During benchmarking, auto-invoking it on small graphs (< 100 nodes) caused regressions by over-connecting unrelated nodes, which degraded retrieval precision.

**Decision**: Edge inference is library code only -- callers must explicitly instantiate `EdgeInferenceEngine` and call `infer_edges()`. It is never auto-invoked during normal event ingestion or plan execution. The executor's orphan detection (`executor/runner.py`, `PlanExecutor._detect_orphan_nodes`) handles the common case of reconnecting orphaned concept/intent nodes via their deterministic `grounded_in` references, creating GROUNDS edges without similarity computation.

**Consequences**:
- Users who want dense similarity-based edges must explicitly call `EdgeInferenceEngine`.
- Small-graph benchmarks remain stable (no spurious over-connection).
- Large-graph scenarios (e.g. bulk wiki ingestion) can benefit from explicit edge inference calls.
- The orphan detection in the executor remains the default safety net for disconnected nodes produced by faulty LLM plans.
