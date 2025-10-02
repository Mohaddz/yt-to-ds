"""
Groq Whisper transcription implementation for YTDS.
"""
import os
import logging
import tempfile
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

from .base import BaseTranscriber
from ..utils import split_audio_into_chunks

logger = logging.getLogger(__name__)


class GroqTranscriber(BaseTranscriber):
    """Groq Whisper transcription service implementation (OpenAI-compatible API)."""
    
    def __init__(self, api_key: str, model: str = "whisper-large-v3", max_workers: int = 6):
        """
        Initialize the Groq transcriber.
        
        Args:
            api_key: Groq API key
            model: Groq Whisper model to use (default: whisper-large-v3)
            max_workers: Maximum number of parallel transcription workers
        """
        super().__init__(api_key)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
        self.max_workers = max_workers
    
    def transcribe_chunk(self, chunk_file: str, chunk_start_ms: int, pbar=None) -> List[Dict]:
        """
        Transcribe a single audio chunk using Groq Whisper API.
        
        Args:
            chunk_file: Path to the audio chunk file
            chunk_start_ms: Start time of the chunk in milliseconds
            pbar: Optional progress bar to update
            
        Returns:
            List of segments with start, end times, and text
        """
        try:
            with open(chunk_file, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Convert response to dictionary if it's not already
            if not isinstance(response, dict):
                response = response.model_dump()
            
            # Add chunk_start_ms to all timestamps
            segments = []
            for segment in response.get("segments", []):
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
        Transcribe audio with timestamps using Groq Whisper API with parallel processing.
        
        Args:
            audio_file: Path to the audio file
            chunk_minutes: Minutes per chunk for long audio
            max_minutes: Maximum minutes to transcribe
            skip_minutes: Minutes to skip from the beginning
            
        Returns:
            List of segments with start, end times, and text
        """
        logger.info(f"Transcribing audio with Groq Whisper: {os.path.basename(audio_file)}")
        
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
            
            # Transcribe chunks in parallel
            all_segments = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all transcription tasks
                with tqdm(total=len(audio_chunks), desc="Transcribing chunks", unit="chunk") as pbar:
                    future_to_chunk = {
                        executor.submit(self.transcribe_chunk, chunk["file"], chunk["start_ms"], pbar): chunk
                        for chunk in audio_chunks
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        try:
                            segments = future.result()
                            all_segments.extend(segments)
                            logger.info(f"Completed transcription of chunk starting at {chunk['start_ms']}ms")
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Chunk transcription failed: {e}")
                            raise
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            logger.info(f"Transcription complete: {len(all_segments)} segments")
            return all_segments

