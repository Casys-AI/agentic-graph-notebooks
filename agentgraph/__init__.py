"""Turn agent execution traces into a graph and measure whether it means anything."""
from .extract import build_hyperedges, iter_tool_calls, mask_secrets
from .graph import metrics, motifs, granularity_sweep
from .stability import temporal_stability, describe_windows

__version__ = "0.1.0"
__all__ = ["build_hyperedges", "iter_tool_calls", "mask_secrets", "metrics",
           "motifs", "granularity_sweep", "temporal_stability", "describe_windows"]
