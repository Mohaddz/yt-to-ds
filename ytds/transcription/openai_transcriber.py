"""
OpenAI Whisper transcription implementation for YTDS.
"""
import os
import json
import logging
import tempfile
from typing import List, Dict, Optional

import openai

from .base import BaseTranscriber
from ..utils import split_audio_into_chunks

logger = logging.getLogger(__name__)


class OpenAITranscriber(BaseTranscriber):
    """OpenAI Whisper transcription service implementation."""
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenAI transcriber.
        
        Args:
            api_key: OpenAI API key
        """
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=api_key)
    
    def transcribe_chunk(self, chunk_file: str, chunk_start_ms: int) -> List[Dict]:
        """
        Transcribe a single audio chunk using OpenAI Whisper API.
        
        Args:
            chunk_file: Path to the audio chunk file
            chunk_start_ms: Start time of the chunk in milliseconds
            
        Returns:
            List of segments with start, end times, and text
        """
        logger.info(f"Transcribing chunk: {os.path.basename(chunk_file)}")
        
        try:
            with open(chunk_file, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Convert response to dictionary if it's not already
            if not isinstance(response, dict):
                response = response.model_dump()
            
            # Add chunk_start_ms to all timestamps
            segments = []
            for segment in response["segments"]:
                segments.append({
                    "start": segment["start"] + (chunk_start_ms / 1000),
                    "end": segment["end"] + (chunk_start_ms / 1000),
                    "text": segment["text"].strip()
                })
            
            return segments
        except Exception as e:
            logger.error(f"Error transcribing chunk {os.path.basename(chunk_file)}: {e}")
            raise
    
    def transcribe_with_timestamps(
        self,
        audio_file: str,
        chunk_minutes: int = 10,
        max_minutes: int = None,
        skip_minutes: int = 0
    ) -> List[Dict]:
        """
        Transcribe audio with timestamps using OpenAI Whisper API.
        
        Args:
            audio_file: Path to the audio file
            chunk_minutes: Minutes per chunk for long audio
            max_minutes: Maximum minutes to transcribe
            skip_minutes: Minutes to skip from the beginning
            
        Returns:
            List of segments with start, end times, and text
        """
        logger.info(f"Transcribing audio with OpenAI Whisper: {os.path.basename(audio_file)}")
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split audio into chunks
            audio_chunks = split_audio_into_chunks(
                audio_file,
                chunk_minutes=chunk_minutes,
                output_dir=temp_dir,
                max_minutes=max_minutes,
                skip_minutes=skip_minutes
            )
            
            # Transcribe each chunk
            all_segments = []
            for chunk in audio_chunks:
                chunk_start_ms = chunk["start_ms"]
                segments = self.transcribe_chunk(chunk["file"], chunk_start_ms)
                all_segments.extend(segments)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            return all_segments
