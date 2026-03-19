from __future__ import annotations

from .inference import DEFAULT_V10_INFERENCE_SNAPSHOT, materialize_inference_snapshot

__all__ = [
    "DEFAULT_V10_INFERENCE_SNAPSHOT",
    "V10BracketPortfolioEngine",
    "materialize_inference_snapshot",
    "write_outputs",
]


def __getattr__(name: str):
    if name == "V10BracketPortfolioEngine":
        from .engine import V10BracketPortfolioEngine

        return V10BracketPortfolioEngine
    if name == "write_outputs":
        from .reporting import write_outputs

        return write_outputs
    raise AttributeError(name)
