from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelManager:
    """Shared model registry.

    Heavy models should be loaded once here and reused across map switches and
    robot sessions.
    """

    _models: Dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, model: Any) -> None:
        self._models[name] = model

    def get(self, name: str) -> Any:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model '{name}' is not registered") from exc
