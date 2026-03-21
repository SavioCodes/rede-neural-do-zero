"""Wrapper package para exposicao mais amigavel no PyPI."""

from src import *  # noqa: F401,F403
from src import __author__, __version__  # noqa: F401

__all__ = [name for name in globals() if not name.startswith("_")]
