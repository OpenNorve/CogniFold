# CogniFold Integrations

CogniFold can plug into external tools as a persistent memory layer. This page
documents the supported integrations and roadmap.

## MCP (Model Context Protocol)

CogniFold ships an MCP server that exposes its persistent concept-graph memory
to any MCP client — Claude Code, Claude Desktop, Cursor, and others. The server
wraps CogniFold's existing in-process logic (the ingestion `Pipeline` and the
`MemoryQueryAgent`); it does not reimplement memory.

### Install

```bash
pip install 'cognifold[mcp]'
```

This pulls in the official `mcp` Python SDK alongside CogniFold. For LLM-based
concept folding you also want an API key and (optionally) the agent extra:

```bash
pip install 'cognifold[mcp,agent]'
```

### Tools

| Tool | Description |
| --- | --- |
| `cognifold_remember(text, timestamp?)` | Ingest an observation/fact/event into the persistent graph. Returns the graph deltas (nodes/edges added, concepts created). |
| `cognifold_query(question, max_nodes?)` | Retrieve relevant context from memory for a question. Returns assembled context + supporting nodes. |
| `cognifold_graph_stats()` | Node/edge counts by type (events, concepts, intents, time nodes). |
| `cognifold_list_intents()` | Current intents (goals/desires) with id, status, and description. |

### Running the server

The server speaks MCP over stdio. Either entry point works:

```bash
python -m cognifold.mcp
# or
cognifold-mcp
```

`cognifold-mcp --help` prints usage and the environment variables it reads.

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `COGNIFOLD_MCP_GRAPH` | Path where the graph JSON is persisted (so memory survives restarts). | `~/.cognifold/mcp_graph.json` |
| `COGNIFOLD_MODEL__NAME` | LLM model used for concept folding during `remember`. | `gemini-2.5-flash` |
| `GOOGLE_API_KEY` | API key for Gemini models. | — |
| `OPENAI_API_KEY` | API key for OpenAI models. | — |

If no API key is set, `cognifold_remember` still works — it falls back to a
default plan that stores the raw event as a node (no LLM-based concept
extraction).

### Claude Desktop

Add CogniFold to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cognifold": {
      "command": "cognifold-mcp",
      "env": {
        "COGNIFOLD_MCP_GRAPH": "/Users/you/.cognifold/mcp_graph.json",
        "COGNIFOLD_MODEL__NAME": "gemini-2.5-flash",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

If `cognifold-mcp` is not on Claude Desktop's `PATH`, use the module form with
an explicit interpreter:

```json
{
  "mcpServers": {
    "cognifold": {
      "command": "python",
      "args": ["-m", "cognifold.mcp"],
      "env": {
        "COGNIFOLD_MCP_GRAPH": "/Users/you/.cognifold/mcp_graph.json",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop after editing the config.

### Claude Code

Register the server with the CLI:

```bash
claude mcp add cognifold \
  --env COGNIFOLD_MCP_GRAPH=$HOME/.cognifold/mcp_graph.json \
  --env COGNIFOLD_MODEL__NAME=gemini-2.5-flash \
  --env GOOGLE_API_KEY=your-key-here \
  -- cognifold-mcp
```

Or commit a project-scoped `.mcp.json` to the repo root:

```json
{
  "mcpServers": {
    "cognifold": {
      "command": "cognifold-mcp",
      "env": {
        "COGNIFOLD_MCP_GRAPH": ".cognifold/mcp_graph.json",
        "COGNIFOLD_MODEL__NAME": "gemini-2.5-flash",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Cursor

Cursor reads the same MCP server schema. Add to
`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project):

```json
{
  "mcpServers": {
    "cognifold": {
      "command": "cognifold-mcp",
      "env": {
        "COGNIFOLD_MCP_GRAPH": "/Users/you/.cognifold/mcp_graph.json",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Example round trip

Once the server is registered, the client's model can call the tools:

```
> cognifold_remember(text="Started using CogniFold as a memory layer for my
                           research notes on 2026-06-18.")

{
  "event_id": "evt-1a2b3c4d5e6f",
  "success": true,
  "nodes_added": 3,
  "edges_added": 2,
  "concepts_created": ["cognifold-memory-layer", "research-notes"],
  "total_nodes": 3,
  "total_edges": 2,
  "graph_path": "/Users/you/.cognifold/mcp_graph.json"
}

> cognifold_query(question="What am I using CogniFold for?")

{
  "question": "What am I using CogniFold for?",
  "context": "CONCEPTS:\n- CogniFold memory layer: used for research notes ...",
  "supporting_nodes": [
    {"node_id": "...", "type": "concept", "title": "CogniFold memory layer",
     "relevance": 0.91, "description": "Memory layer for research notes"}
  ],
  "nodes_scanned": 3,
  "query_time_ms": 4.2
}
```

The memory persists at `COGNIFOLD_MCP_GRAPH`, so a follow-up `cognifold_query`
in a later session still recalls these facts.

---

## Roadmap (Planned)

The following integrations are **planned** and not yet implemented:

- **OpenAI-compatible API** — _Planned._ A drop-in `/v1`-style HTTP surface so
  CogniFold memory can be used by any OpenAI-compatible client/SDK.
- **LangChain `BaseMemory`** — _Planned._ A `BaseMemory` subclass backed by
  CogniFold so LangChain chains/agents can read and write the concept graph.
- **LlamaIndex retriever** — _Planned._ A `BaseRetriever` implementation that
  queries the CogniFold graph for use in LlamaIndex query engines.

Contributions welcome — these wrap the same in-process `Pipeline` /
`MemoryQueryAgent` seams the MCP server already uses.
