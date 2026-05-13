"""Interactive CLI client for the Cognifold HTTP service.

Connects to a running Cognifold server for session management,
event ingestion, querying, and graph exploration.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import urllib.error
import urllib.request
from typing import Any


class _CommandError(Exception):
    """Raised to abort a REPL command with a user-facing message."""


class CognifoldClient:
    """HTTP client for the Cognifold service API (stdlib only)."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        """Send an HTTP request and return (status_code, parsed_json | None)."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req) as resp:
                status: int = resp.status
                raw = resp.read()
                if not raw:
                    return status, None
                return status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw_body = exc.read()
            detail: dict[str, Any] | None = None
            if raw_body:
                try:
                    detail = json.loads(raw_body)
                except (json.JSONDecodeError, ValueError):
                    detail = {"detail": raw_body.decode(errors="replace")}
            return exc.code, detail
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot connect to {self.base_url}: {exc.reason}") from exc

    # --- Health ---

    def health(self) -> dict[str, Any]:
        _, data = self._request("GET", "/health")
        return data or {}

    # --- Sessions ---

    def create_session(
        self,
        llm_api_keys: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if llm_api_keys:
            body["llm_api_keys"] = llm_api_keys
        if config:
            body["config"] = config
        _, data = self._request("POST", "/api/v1/sessions", body=body)
        return data or {}

    def get_session(self, session_id: str) -> dict[str, Any]:
        status, data = self._request("GET", f"/api/v1/sessions/{session_id}")
        if status == 404:
            raise _CommandError(f"Session not found: {session_id}")
        return data or {}

    def delete_session(self, session_id: str) -> None:
        status, data = self._request("DELETE", f"/api/v1/sessions/{session_id}")
        if status == 404:
            raise _CommandError(f"Session not found: {session_id}")
        if status >= 400:
            detail = (data or {}).get("detail", "unknown error")
            raise _CommandError(f"Failed to delete session: {detail}")

    def load_graph(self, session_id: str, graph_path: str) -> dict[str, Any]:
        with open(graph_path) as f:
            graph_data = json.load(f)
        _, data = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/load",
            body={"graph_data": graph_data},
        )
        return data or {}

    # --- Events ---

    def ingest_event(
        self,
        session_id: str,
        event_type: str,
        title: str,
        description: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {"event_type": event_type, "title": title}
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        _, data = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/events",
            body={"event": event},
        )
        return data or {}

    # --- Query ---

    def query(self, session_id: str, query_text: str) -> dict[str, Any]:
        _, data = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/query",
            body={"query": query_text},
        )
        return data or {}

    # --- Graph ---

    def get_graph_stats(self, session_id: str) -> dict[str, Any]:
        _, data = self._request("GET", f"/api/v1/sessions/{session_id}/graph/stats")
        return data or {}

    def get_graph_state(self, session_id: str, max_nodes: int = 20) -> dict[str, Any]:
        _, data = self._request("GET", f"/api/v1/sessions/{session_id}/graph?max_nodes={max_nodes}")
        return data or {}

    def get_top_concepts(self, session_id: str, top: int = 10) -> list[dict[str, Any]]:
        _, data = self._request("GET", f"/api/v1/sessions/{session_id}/graph/concepts?top={top}")
        return data if isinstance(data, list) else []

    def get_recent_intents(self, session_id: str, recent: int = 10) -> list[dict[str, Any]]:
        _, data = self._request(
            "GET", f"/api/v1/sessions/{session_id}/graph/intents?recent={recent}"
        )
        return data if isinstance(data, list) else []

    # --- Intent Personalization (Phase 14.1) ---

    def submit_feedback(
        self,
        session_id: str,
        intent_id: str,
        feedback_type: str,
        user_comment: str | None = None,
        modified_priority: str | None = None,
        modified_description: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"feedback_type": feedback_type}
        if user_comment:
            body["user_comment"] = user_comment
        if modified_priority:
            body["modified_priority"] = modified_priority
        if modified_description:
            body["modified_description"] = modified_description
        status, data = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/intents/{intent_id}/feedback",
            body=body,
        )
        if status == 404:
            raise _CommandError(data.get("detail", "not found") if data else "not found")
        return data or {}

    def get_calibration(self, session_id: str) -> dict[str, Any]:
        _, data = self._request("GET", f"/api/v1/sessions/{session_id}/intents/calibration")
        return data or {}

    def get_pending_intents(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        _, data = self._request(
            "GET", f"/api/v1/sessions/{session_id}/intents/pending?limit={limit}"
        )
        return data if isinstance(data, list) else []

    def get_recent_events(self, session_id: str, recent: int = 10) -> list[dict[str, Any]]:
        _, data = self._request(
            "GET", f"/api/v1/sessions/{session_id}/graph/events?recent={recent}"
        )
        return data if isinstance(data, list) else []

    def get_node(self, session_id: str, node_id: str) -> dict[str, Any]:
        status, data = self._request("GET", f"/api/v1/sessions/{session_id}/graph/nodes/{node_id}")
        if status == 404:
            raise _CommandError(f"Node not found: {node_id}")
        return data or {}

    def expand_node(
        self,
        session_id: str,
        node_id: str,
        layers: int = 1,
        direction: str = "both",
        max_nodes: int = 50,
    ) -> dict[str, Any]:
        params = f"layers={layers}&direction={direction}&max_nodes={max_nodes}"
        status, data = self._request(
            "GET",
            f"/api/v1/sessions/{session_id}/graph/nodes/{node_id}/expand?{params}",
        )
        if status == 404:
            detail = (data or {}).get("detail", "not found")
            raise _CommandError(detail)
        return data or {}


class ClientREPL:
    """Interactive REPL for the Cognifold service."""

    def __init__(
        self,
        client: CognifoldClient,
        session_id: str | None = None,
        llm_api_keys: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.llm_api_keys = llm_api_keys or {}
        self.model_name = model_name

    @property
    def prompt(self) -> str:
        if self.session_id:
            short = self.session_id[:8]
            return f"cognifold [{short}]> "
        return "cognifold> "

    def _require_session(self) -> str:
        if not self.session_id:
            raise _CommandError("No active session. Use :session create")
        return self.session_id

    # --- Command handlers ---

    def _cmd_help(self) -> None:
        print("Commands:")
        print("  :help / :h               Show this help")
        print("  :connect [URL]           Set server URL, verify with /health")
        print("  :session create          Create a new session")
        print("  :session info            Show current session info")
        print("  :session delete          Delete current session")
        print("  :session <ID>            Switch to existing session")
        print("  :stats                   Show graph statistics")
        print("  :concepts [N]            Top N concepts (default 10)")
        print("  :intents [N]             Recent N intents (default 10)")
        print("  :intents pending [N]     Pending intents with calibration scores")
        print("  :feedback <ID> TYPE [comment]  Submit intent feedback")
        print("                           TYPE: accept | reject | defer | modify")
        print("  :calibration             Show intent calibration profile")
        print("  :events [N]              Recent N events (default 10)")
        print("  :node <ID>               Show node details")
        print("  :expand <ID> [N] [--direction D] [--max M]")
        print("                           Expand from node by N layers (default 1)")
        print("  :graph [N]               Show graph state (default 20 nodes)")
        print("  :load <FILE>             Load graph JSON into session")
        print("  :ingest TYPE TITLE [--desc D] [--loc L]  Ingest an event")
        print("  :quit / :q / :exit       Exit")
        print()
        print("Anything without ':' is treated as a natural language query.")

    def _cmd_connect(self, arg: str) -> None:
        if arg:
            self.client = CognifoldClient(arg, api_key=self.client.api_key)
        result = self.client.health()
        status = result.get("status", "unknown")
        print(f"Connected to {self.client.base_url} (status: {status})")

    def _cmd_session(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :session create | info | delete | <ID>")

        sub = arg.split(maxsplit=1)[0].lower()

        if sub == "create":
            config: dict[str, Any] | None = None
            if self.model_name:
                config = {"model_name": self.model_name}
            info = self.client.create_session(
                llm_api_keys=self.llm_api_keys or None,
                config=config,
            )
            self.session_id = info.get("session_id", "")
            print(f"Created session: {self.session_id}")
            if self.llm_api_keys:
                providers = ", ".join(self.llm_api_keys.keys())
                print(f"  LLM keys: {providers}")
            if self.model_name:
                print(f"  Model: {self.model_name}")
            stats = info.get("graph_stats")
            if stats:
                print(f"  Nodes: {stats.get('node_count', 0)}, Edges: {stats.get('edge_count', 0)}")

        elif sub == "info":
            sid = self._require_session()
            info = self.client.get_session(sid)
            print(f"Session:  {info.get('session_id', sid)}")
            print(f"Created:  {info.get('created_at', '?')}")
            print(f"Accessed: {info.get('last_accessed', '?')}")
            config = info.get("config", {})
            if config:
                print(f"Model:    {config.get('model_name', '?')}")
                print(f"Domain:   {config.get('domain', '?')}")
            stats = info.get("graph_stats")
            if stats:
                print(f"Nodes:    {stats.get('node_count', 0)}")
                print(f"Edges:    {stats.get('edge_count', 0)}")

        elif sub == "delete":
            sid = self._require_session()
            self.client.delete_session(sid)
            print(f"Deleted session: {sid}")
            self.session_id = None

        else:
            # Treat as session ID to switch to
            session_id = sub
            info = self.client.get_session(session_id)
            self.session_id = info.get("session_id", session_id)
            print(f"Switched to session: {self.session_id}")

    def _cmd_stats(self) -> None:
        sid = self._require_session()
        stats = self.client.get_graph_stats(sid)
        print(f"Nodes: {stats.get('node_count', 0)}  Edges: {stats.get('edge_count', 0)}")
        print(f"  Concepts: {stats.get('concepts', 0)}")
        print(f"  Events:   {stats.get('events', 0)}")
        print(f"  Intents:  {stats.get('intents', 0)}")
        print(f"  Time:     {stats.get('time_nodes', 0)}")

    def _cmd_concepts(self, arg: str) -> None:
        sid = self._require_session()
        n = int(arg) if arg else 10
        items = self.client.get_top_concepts(sid, top=n)
        if not items:
            print("No concepts found.")
            return
        print(f"Top {len(items)} Concepts:")
        for i, c in enumerate(items, 1):
            score: float = c.get("relevance_score", 0.0)
            print(f"  {i}. {c.get('title', '?')} (score: {score:.2f})")

    def _cmd_intents(self, arg: str) -> None:
        sid = self._require_session()
        # Sub-command: :intents pending [N]
        parts = arg.split(maxsplit=1) if arg else []
        if parts and parts[0].lower() == "pending":
            n = int(parts[1]) if len(parts) > 1 else 20
            items = self.client.get_pending_intents(sid, limit=n)
            if not items:
                print("No pending intents.")
                return
            print(f"Pending Intents ({len(items)}):")
            for i, item in enumerate(items, 1):
                mult = item.get("score_multiplier", 1.0)
                prio = item.get("priority", "?")
                print(f"  {i}. [{prio}] {item.get('title', '?')} (calibration: {mult:.2f})")
                if item.get("intent_id"):
                    print(f"     ID: {item['intent_id']}")
            return

        n = int(arg) if arg else 10
        items = self.client.get_recent_intents(sid, recent=n)
        if not items:
            print("No intents found.")
            return
        print(f"Recent {len(items)} Intents:")
        for i, item in enumerate(items, 1):
            score: float = item.get("relevance_score", 0.0)
            print(f"  {i}. {item.get('title', '?')} (score: {score:.2f})")

    def _cmd_feedback(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :feedback <intent-id> accept|reject|defer|modify [comment]")
        sid = self._require_session()

        try:
            tokens = shlex.split(arg)
        except ValueError as exc:
            raise _CommandError(f"Parse error: {exc}") from exc

        if len(tokens) < 2:
            raise _CommandError("Usage: :feedback <intent-id> accept|reject|defer|modify [comment]")

        intent_id = tokens[0]
        feedback_type = tokens[1].lower()
        valid_types = ("accept", "reject", "defer", "modify")
        if feedback_type not in valid_types:
            raise _CommandError(f"Invalid type: {feedback_type}. Must be one of {valid_types}")

        comment = " ".join(tokens[2:]) if len(tokens) > 2 else None

        result = self.client.submit_feedback(sid, intent_id, feedback_type, user_comment=comment)
        fb_id = result.get("feedback_id", "?")
        status = result.get("intent_status", "?")
        print(f"Feedback recorded: {fb_id}")
        print(f"  Type: {feedback_type}, Intent status: {status}")

    def _cmd_calibration(self) -> None:
        sid = self._require_session()
        data = self.client.get_calibration(sid)
        total = data.get("total_feedback", 0)
        rate = data.get("acceptance_rate", 0.0)
        print(f"Calibration Profile ({total} feedback entries)")
        print(f"  Acceptance rate: {rate:.0%}")
        print(f"  Priority bias:   {data.get('priority_bias', 0.0):+.2f}")
        cats = data.get("category_weights", {})
        if cats:
            print("  Category weights:")
            for cat, w in sorted(cats.items(), key=lambda x: -x[1]):
                label = "preferred" if w > 1.1 else ("disliked" if w < 0.9 else "neutral")
                print(f"    {cat}: {w:.2f} ({label})")
        rejected = data.get("rejection_patterns", [])
        if rejected:
            print(f"  Rejection patterns: {', '.join(rejected)}")
        preferred = data.get("preferred_patterns", [])
        if preferred:
            print(f"  Preferred patterns: {', '.join(preferred)}")

    def _cmd_events(self, arg: str) -> None:
        sid = self._require_session()
        n = int(arg) if arg else 10
        items = self.client.get_recent_events(sid, recent=n)
        if not items:
            print("No events found.")
            return
        print(f"Recent {len(items)} Events:")
        for i, item in enumerate(items, 1):
            ntype: str = item.get("node_type", "")
            print(f"  {i}. [{ntype}] {item.get('title', '?')}")

    def _cmd_node(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :node <ID>")
        sid = self._require_session()
        node = self.client.get_node(sid, arg)
        print(f"Node:     {node.get('node_id', arg)}")
        print(f"Type:     {node.get('node_type', '?')}")
        data = node.get("data", {})
        print(f"Title:    {data.get('title', '?')}")
        if data.get("description"):
            print(f"Desc:     {data['description']}")
        print(f"Created:  {node.get('created_at', '?')}")
        print(f"Accessed: {node.get('last_accessed', '?')} ({node.get('access_count', 0)}x)")
        if node.get("reasoning"):
            print(f"Reason:   {node['reasoning']}")
        neighbors = node.get("neighbors", [])
        if neighbors:
            print(f"Neighbors:    {', '.join(neighbors)}")
        predecessors = node.get("predecessors", [])
        if predecessors:
            print(f"Predecessors: {', '.join(predecessors)}")

    def _cmd_expand(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :expand <ID> [LAYERS] [--direction D] [--max M]")
        sid = self._require_session()

        try:
            tokens = shlex.split(arg)
        except ValueError as exc:
            raise _CommandError(f"Parse error: {exc}") from exc

        node_id = tokens[0]
        layers = 1
        direction = "both"
        max_nodes = 50

        i = 1
        while i < len(tokens):
            if tokens[i] == "--direction" and i + 1 < len(tokens):
                direction = tokens[i + 1]
                if direction not in ("outgoing", "incoming", "both"):
                    raise _CommandError("direction must be outgoing, incoming, or both")
                i += 2
            elif tokens[i] == "--max" and i + 1 < len(tokens):
                max_nodes = int(tokens[i + 1])
                i += 2
            elif tokens[i].isdigit():
                layers = int(tokens[i])
                i += 1
            else:
                raise _CommandError(f"Unknown option: {tokens[i]}")

        data = self.client.expand_node(
            sid, node_id, layers=layers, direction=direction, max_nodes=max_nodes
        )
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        truncated = data.get("truncated", False)

        print(f"Expansion from {node_id} ({layers} layer(s), {direction}):")
        print(f"  {len(nodes)} nodes, {len(edges)} edges{' (truncated)' if truncated else ''}")
        print()

        # Group nodes by depth
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for n in nodes:
            depth = n.get("depth", 0)
            by_depth.setdefault(depth, []).append(n)

        for depth in sorted(by_depth):
            label = "Root" if depth == 0 else f"Depth {depth}"
            print(f"  {label}:")
            for n in by_depth[depth]:
                nid = n.get("node_id", "?")
                ntype = n.get("node_type", "?")
                title = n.get("data", {}).get("title", "?")
                print(f"    [{ntype}] {nid[:20]}  {title}")

        if edges:
            print()
            print("  Edges:")
            for e in edges:
                src = e.get("source_id", "?")[:12]
                tgt = e.get("target_id", "?")[:12]
                etype = e.get("edge_type") or "—"
                weight = e.get("weight", 1.0)
                print(f"    {src} —[{etype} {weight:.1f}]→ {tgt}")

    def _cmd_graph(self, arg: str) -> None:
        sid = self._require_session()
        n = int(arg) if arg else 20
        data = self.client.get_graph_state(sid, max_nodes=n)
        stats = data.get("stats", {})
        print(f"Graph: {stats.get('node_count', 0)} nodes, {stats.get('edge_count', 0)} edges")
        nodes = data.get("nodes", [])
        for node in nodes:
            nid = node.get("node_id", "?")
            ntype = node.get("node_type", "?")
            title = node.get("data", {}).get("title", "?")
            print(f"  [{ntype}] {nid[:12]}  {title}")

    def _cmd_load(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :load <FILE>")
        sid = self._require_session()
        info = self.client.load_graph(sid, arg)
        stats = info.get("graph_stats", {})
        print(f"Graph loaded into session {sid}")
        print(f"  Nodes: {stats.get('node_count', 0)}, Edges: {stats.get('edge_count', 0)}")

    def _cmd_ingest(self, arg: str) -> None:
        if not arg:
            raise _CommandError("Usage: :ingest TYPE TITLE [--desc DESC] [--loc LOC]")
        sid = self._require_session()

        try:
            tokens = shlex.split(arg)
        except ValueError as exc:
            raise _CommandError(f"Parse error: {exc}") from exc

        if len(tokens) < 2:
            raise _CommandError("Usage: :ingest TYPE TITLE [--desc DESC] [--loc LOC]")

        event_type = tokens[0]
        title = tokens[1]
        description: str | None = None
        location: str | None = None

        i = 2
        while i < len(tokens):
            if tokens[i] == "--desc" and i + 1 < len(tokens):
                description = tokens[i + 1]
                i += 2
            elif tokens[i] == "--loc" and i + 1 < len(tokens):
                location = tokens[i + 1]
                i += 2
            else:
                raise _CommandError(f"Unknown option: {tokens[i]}")

        result = self.client.ingest_event(
            sid, event_type, title, description=description, location=location
        )
        success = result.get("success", False)
        ops = result.get("operations_completed", 0)
        ms = result.get("execution_time_ms", 0.0)
        print(f"{'OK' if success else 'FAILED'}: {ops} operations in {ms:.0f}ms")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        if result.get("reasoning"):
            print(f"  Reasoning: {result['reasoning']}")

    def _cmd_query(self, text: str) -> None:
        sid = self._require_session()
        result = self.client.query(sid, text)
        context = result.get("context", "")
        if context:
            print(context)
        else:
            nodes = result.get("nodes", [])
            if nodes:
                for n in nodes:
                    print(
                        f"  [{n.get('node_type', '?')}] {n.get('title', '?')} "
                        f"(score: {n.get('relevance_score', 0):.2f})"
                    )
            else:
                print("No results found.")
        ms = result.get("query_time_ms", 0.0)
        print(f"  ({ms:.0f}ms)")

    def dispatch(self, line: str) -> bool:
        """Dispatch a line of input. Returns False to exit the REPL."""
        if not line:
            return True

        if line.startswith(":"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "q", "exit"):
                return False

            handler_map: dict[str, Any] = {
                "help": lambda: self._cmd_help(),
                "h": lambda: self._cmd_help(),
                "connect": lambda: self._cmd_connect(arg),
                "session": lambda: self._cmd_session(arg),
                "stats": lambda: self._cmd_stats(),
                "concepts": lambda: self._cmd_concepts(arg),
                "intents": lambda: self._cmd_intents(arg),
                "feedback": lambda: self._cmd_feedback(arg),
                "calibration": lambda: self._cmd_calibration(),
                "events": lambda: self._cmd_events(arg),
                "node": lambda: self._cmd_node(arg),
                "expand": lambda: self._cmd_expand(arg),
                "graph": lambda: self._cmd_graph(arg),
                "load": lambda: self._cmd_load(arg),
                "ingest": lambda: self._cmd_ingest(arg),
            }

            handler = handler_map.get(cmd)
            if handler is None:
                print(f"Unknown command: {cmd}. Type :help for available commands.")
                return True

            try:
                handler()
            except _CommandError as exc:
                print(f"Error: {exc}")
            except ConnectionError as exc:
                print(f"Connection error: {exc}")
            except Exception as exc:
                print(f"Error: {exc}")
        else:
            # Natural language query
            try:
                self._cmd_query(line)
            except _CommandError as exc:
                print(f"Error: {exc}")
            except ConnectionError as exc:
                print(f"Connection error: {exc}")
            except Exception as exc:
                print(f"Error: {exc}")

        return True

    def run(self) -> int:
        """Run the interactive REPL loop."""
        print("=" * 50)
        print("  Cognifold Interactive Client")
        print("=" * 50)
        print(f"  Server: {self.client.base_url}")
        if self.session_id:
            print(f"  Session: {self.session_id}")
        if self.llm_api_keys:
            providers = ", ".join(self.llm_api_keys.keys())
            print(f"  LLM:    {providers}")
        else:
            print("  LLM:    none (events use default plan)")
        print("  Type :help for commands, or enter a query.")
        print()

        while True:
            try:
                line = input(self.prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return 0

            if not self.dispatch(line):
                print("Goodbye.")
                return 0


# --- CLI registration ---


def add_client_parser(subparsers: Any) -> None:
    """Add the 'client' subcommand parser."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "client",
        help="Interactive client for the Cognifold service",
        description="Connect to a running Cognifold server for session management, "
        "event ingestion, querying, and graph exploration.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (also reads COGNIFOLD_API_KEY env var)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Connect to an existing session by ID",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for LLM agent (also reads OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--google-api-key",
        default=None,
        help="Google API key for LLM agent (also reads GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (e.g. gemini-2.5-flash, gemini-3-flash-preview)",
    )


def client_command(args: argparse.Namespace) -> int:
    """Run the interactive client."""
    api_key = args.api_key or os.environ.get("COGNIFOLD_API_KEY")
    client = CognifoldClient(args.url, api_key=api_key)

    # Collect LLM API keys from flags and env vars
    llm_api_keys: dict[str, str] = {}
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    google_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
    if openai_key:
        llm_api_keys["openai"] = openai_key
    if google_key:
        llm_api_keys["google"] = google_key

    repl = ClientREPL(
        client,
        session_id=args.session,
        llm_api_keys=llm_api_keys,
        model_name=args.model,
    )
    return repl.run()
