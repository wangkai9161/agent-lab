"""Unified model factory.

Every trainable model file in this package should expose a ``rec_model()``
factory. Training and testing scripts call ``create_model(name)`` instead of
importing model modules directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List


_EXCLUDED_MODULES = {"__init__"}


def available_models() -> List[str]:
    """Return model module names that can be selected by the main scripts."""
    model_dir = Path(__file__).resolve().parent
    names = []
    for path in model_dir.glob("*.py"):
        if path.stem in _EXCLUDED_MODULES:
            continue
        names.append(path.stem)
    return sorted(names, key=str.lower)


def resolve_model_name(name: str) -> str:
    """Resolve a user-provided model name to the exact module filename stem."""
    candidates = available_models()
    if name in candidates:
        return name

    lowered = name.lower()
    for candidate in candidates:
        if candidate.lower() == lowered:
            return candidate

    raise ValueError(
        f"Unknown model '{name}'. Available models: {', '.join(candidates)}"
    )


def create_model(name: str):
    """Instantiate a model by module name using its ``rec_model()`` factory."""
    import torch.nn as nn

    module_name = resolve_model_name(name)
    module = importlib.import_module(f"{__name__}.{module_name}")

    if not hasattr(module, "rec_model"):
        raise AttributeError(
            f"models.{module_name} must define rec_model() to be used by the main scripts."
        )

    model = module.rec_model()
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"models.{module_name}.rec_model() must return torch.nn.Module, got {type(model)!r}."
        )
    return model
