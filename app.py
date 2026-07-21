"""Backward-compatible ASGI entry point.

New deployments should use ``src.api.main:app``.
"""

from src.api.main import app

__all__ = ["app"]
