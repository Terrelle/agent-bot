"""Task registry for execution agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


# Return tool schemas contributed by task modules
def get_task_schemas() -> List[Dict[str, Any]]:
    """Return tool schemas contributed by task modules."""
    return []


# Return executable task tools keyed by name
def get_task_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:
    """Return executable task tools keyed by name."""
    return {}


__all__ = [
    "get_task_registry",
    "get_task_schemas",
]
