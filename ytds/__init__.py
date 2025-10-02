"""
YTDS - YouTube to Dataset Library

A library for converting YouTube videos to transcribed datasets.
"""

__version__ = "0.1.0"

from .ytds import YTDSProcessor
from .cli import main

__all__ = ["YTDSProcessor", "main"]
