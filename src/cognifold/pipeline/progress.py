"""Progress reporting for layered pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LayerProgress:
    """Progress state for a single pipeline layer."""

    layer: int
    label: str
    total: int = 0
    completed: int = 0
    errors: int = 0
    elapsed_ms: float = 0.0

    @property
    def pct(self) -> float:
        """Completion percentage (0-100)."""
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100.0


class ProgressCallback(Protocol):
    """Protocol for receiving pipeline progress updates."""

    def on_layer_start(self, progress: LayerProgress) -> None: ...
    def on_layer_progress(self, progress: LayerProgress) -> None: ...
    def on_layer_complete(self, progress: LayerProgress) -> None: ...


class PrintProgressCallback:
    """Simple progress callback that prints to stdout."""

    def on_layer_start(self, progress: LayerProgress) -> None:
        print(f"\n[Layer {progress.layer}] {progress.label} ({progress.total} items)")

    def on_layer_progress(self, progress: LayerProgress) -> None:
        print(
            f"  [{progress.completed}/{progress.total}] "
            f"{progress.pct:.0f}% ({progress.elapsed_ms:.0f}ms)",
            end="\r",
        )

    def on_layer_complete(self, progress: LayerProgress) -> None:
        print(
            f"  [Layer {progress.layer}] Done — "
            f"{progress.completed} ok, {progress.errors} errors, "
            f"{progress.elapsed_ms:.0f}ms"
        )


@dataclass
class FastPipelineStats:
    """Aggregate stats across all layers of a fast pipeline run."""

    layer1_events: int = 0
    layer1_time_ms: float = 0.0
    layer2_batches: int = 0
    layer2_plans: int = 0
    layer2_time_ms: float = 0.0
    layer3_nodes_embedded: int = 0
    layer3_time_ms: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        return self.layer1_time_ms + self.layer2_time_ms + self.layer3_time_ms
