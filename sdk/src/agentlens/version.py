"""Single source of truth for the SDK version.

Kept in its own module so ``pyproject.toml`` metadata, the resource attribute
``agentlens.sdk.version``, and the evaluator/pricing provenance records cannot
drift apart.
"""

__version__ = "0.1.0"
