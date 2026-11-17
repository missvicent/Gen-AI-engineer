"""Build the model-facing tool spec from the function itself.

The model sees only what this returns: the function's name, its docstring as the
description, and its type hints as the parameters. So the docstring IS the
interface — blank it (blank=True) and the model gets an empty description.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

# Python type -> JSON Schema type. Only what our tools use; add as needed.
_JSON_TYPES: dict[type, str] = {str: "string", int: "integer", bool: "boolean"}


def spec_from_function(fn: Callable[..., Any], blank: bool = False) -> dict[str, Any]:
    hints = get_type_hints(fn)  # resolves the string hints `from __future__` makes
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        properties[name] = {"type": _JSON_TYPES[hints[name]]}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": "" if blank else inspect.getdoc(fn) or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
