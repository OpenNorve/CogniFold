import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath("src"))

from cognifold.replay.renderer import ReplayRenderer
from cognifold.replay.player import ReplayPlayer, Keyframe

def generate_demo():
    print("Generating demo replay...")
    
    # Mock Nodes
    nodes = [
        {"id": "e1", "type": "event", "data": {"title": "User Login", "description": "User logged into the system"}},
        {"id": "c1", "type": "concept", "data": {"title": "Authentication", "strength": 0.95}, "reasoning": "Core concept triggered by login"},
        {"id": "i1", "type": "intent", "data": {"title": "Verify Credentials", "priority": "high", "status": "pending"}},
        {"id": "t1", "type": "time", "data": {"title": "2023-10-27 10:00:00"}},
    ]
    
    edges = [
        {"source": "e1", "target": "c1"},
        {"source": "c1", "target": "i1"},
        {"source": "e1", "target": "t1"},
    ]
    
    # Mock Keyframe
    kf = Keyframe(
        step=1,
        event_id="e1",
        event_title="User Login Event",
        event_type="event",
        nodes=nodes,
        edges=edges,
        context_node_ids=["e1", "c1"],
        scores={"e1": 1.0, "c1": 0.9, "i1": 0.8, "t1": 0.5},
        operations=[
            {"op": "ADD_NODE", "node_type": "event", "data": {"title": "User Login"}},
            {"op": "ADD_EDGE", "source_id": "e1", "target_id": "c1"}
        ],
        reasoning="Processing user login event and activating authentication concepts.",
        added_nodes=["e1", "c1", "i1", "t1"],
        intents_selected=[
            {"intent_id": "i1", "intent_title": "Verify Credentials", "urgency_score": 0.9, "status": "pending"}
        ]
    )
    
    player = ReplayPlayer(
        entries=[],
        keyframes=[kf],
        metadata={"timeline_path": "demo_timeline.json"}
    )
    
    renderer = ReplayRenderer()
    output_path = Path("demo_replay.html")
    renderer.render(player, output_path, title="Cognifold UI Demo")
    
    print(f"Successfully generated demo at: {output_path.absolute()}")

if __name__ == "__main__":
    generate_demo()
