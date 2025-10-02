"""
Base transcriber class for YTDS.
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTranscriber(ABC):
    """Base class for audio transcription services."""
    
    def __init__(self, api_key: str):
        """
        Initialize the transcriber.
        
        Args:
            api_key: API key for the transcription service
        """
        self.api_key = api_key
    
    @abstractmethod
    def transcribe_with_timestamps(
        self,
        audio_file: str,
        chunk_minutes: int = 10,
        max_minutes: int = None,
        skip_minutes: int = 0
    ) -> List[Dict]:
        """
        Transcribe audio with timestamps.
        
        Args:
            audio_file: Path to the audio file
            chunk_minutes: Minutes per chunk for long audio
            max_minutes: Maximum minutes to transcribe
            skip_minutes: Minutes to skip from the beginning
            
        Returns:
            List of segments with start, end times, and text
        """
        pass
