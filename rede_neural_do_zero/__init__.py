"""Wrapper package para exposicao mais amigavel no PyPI.

Mantem a interface publica centralizada em `src`, mas oferece um nome de
pacote mais claro para instalacao, imports e documentacao oficial.
"""

from src import *  # noqa: F401,F403
from src import __author__, __version__  # noqa: F401

__all__ = [name for name in globals() if not name.startswith("_")]
