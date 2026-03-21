"""Backward-compatible exports for benchmark helpers.

The project is now organized in subpackages. This module remains as a
stable import path for users and tests that still import from `src.benchmarking`.
"""

from .workflows.benchmarking import *  # noqa: F401,F403
