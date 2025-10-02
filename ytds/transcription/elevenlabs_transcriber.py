"""
ElevenLabs transcription implementation for YTDS.
"""
import os
import json
import logging
import tempfile
import time
import requests
from typing import List, Dict, Optional

from tqdm import tqdm

from .base import BaseTranscriber
from ..utils import split_audio_into_chunks

logger = logging.getLogger(__name__)


class ElevenLabsTranscriber(BaseTranscriber):
    """ElevenLabs transcription service implementation."""
    
    def __init__(self, api_key: str):
        """
        Initialize the ElevenLabs transcriber.
        
        Args:
            api_key: ElevenLabs API key
        """
        super().__init__(api_key)
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": api_key,
            "Accept": "application/json"
        }
    
    def transcribe_chunk(self, chunk_file: str, chunk_start_ms: int) -> List[Dict]:
        """
        Transcribe a single audio chunk using ElevenLabs API.
        
        Args:
            chunk_file: Path to the audio chunk file
            chunk_start_ms: Start time of the chunk in milliseconds
            
        Returns:
            List of segments with start, end times, and text
        """
        logger.info(f"Transcribing chunk with ElevenLabs: {os.path.basename(chunk_file)}")
        
        try:
            # ElevenLabs Speech-to-Text API endpoint
            url = f"{self.base_url}/speech-to-text"
            
            # Prepare files and parameters
            files = {
                "audio": (os.path.basename(chunk_file), open(chunk_file, "rb"), "audio/mpeg")
            }
            data = {
                "transcription_config": json.dumps({
                    "chunk_text": True,
                    "language_detection": True,
                    "enable_timestamps": True
                })
            }
            
            # Make API request
            response = requests.post(url, headers=self.headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            # Format the response into segments
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment.get("start", 0) + (chunk_start_ms / 1000),
                    "end": segment.get("end", 0) + (chunk_start_ms / 1000),
                    "text": segment.get("text", "").strip()
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {os.path.basename(chunk_file)} with ElevenLabs: {e}")
            raise
        finally:
            # Close the file opened in files
            if "files" in locals() and "audio" in files:
                files["audio"][1].close()
    
    def transcribe_with_timestamps(
        self,
        audio_file: str,
        chunk_minutes: int = 10,
        max_minutes: int = None,
        skip_minutes: int = 0
    ) -> List[Dict]:
        """
        Transcribe audio with timestamps using ElevenLabs API.
        
        Args:
            audio_file: Path to the audio file
            chunk_minutes: Minutes per chunk for long audio
            max_minutes: Maximum minutes to transcribe
            skip_minutes: Minutes to skip from the beginning
            
        Returns:
            List of segments with start, end times, and text
        """
        logger.info(f"Transcribing audio with ElevenLabs: {os.path.basename(audio_file)}")
        
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
            for chunk in tqdm(audio_chunks, desc="Transcribing chunks", unit="chunk"):
                chunk_start_ms = chunk["start_ms"]
                segments = self.transcribe_chunk(chunk["file"], chunk_start_ms)
                all_segments.extend(segments)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            return all_segments
