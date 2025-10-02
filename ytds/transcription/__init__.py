"""
Transcription module for YTDS.
"""
from .base import BaseTranscriber
from .openai_transcriber import OpenAITranscriber
from .elevenlabs_transcriber import ElevenLabsTranscriber
from .groq_transcriber import GroqTranscriber

__all__ = ["BaseTranscriber", "OpenAITranscriber", "ElevenLabsTranscriber", "GroqTranscriber"]
