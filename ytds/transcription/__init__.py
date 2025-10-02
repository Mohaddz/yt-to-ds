"""
Transcription module for YTDS.
"""
from .base import BaseTranscriber
from .openai_transcriber import OpenAITranscriber
from .elevenlabs_transcriber import ElevenLabsTranscriber

__all__ = ["BaseTranscriber", "OpenAITranscriber", "ElevenLabsTranscriber"]
