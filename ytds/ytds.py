"""
Main module for YTDS (YouTube to Dataset) processor.
"""
import os
import sys
import logging
import tempfile
from typing import List, Dict, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YTDSProcessor:
    """Main processor class for converting YouTube videos to datasets."""
    
    def __init__(
        self, 
        openai_api_key: str = None,
        elevenlabs_api_key: str = None,
        groq_api_key: str = None,
        hf_token: str = None,
        ffmpeg_path: str = None,
        output_dir: str = None,
        transcription_provider: str = "openai",
        max_workers: int = 6
    ):
        """
        Initialize the YTDS processor.
        
        Args:
            openai_api_key: OpenAI API key for Whisper transcription
            elevenlabs_api_key: ElevenLabs API key for transcription
            groq_api_key: Groq API key for Whisper transcription
            hf_token: HuggingFace token for dataset upload
            ffmpeg_path: Custom path to ffmpeg binary
            output_dir: Directory to store outputs (default: temporary directory)
            transcription_provider: Provider for transcription ("openai", "elevenlabs", or "groq")
            max_workers: Maximum number of parallel workers for transcription (default: 6)
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.elevenlabs_api_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.ffmpeg_path = ffmpeg_path
        self.transcription_provider = transcription_provider
        self.max_workers = max_workers
        
        # Validate provider
        if transcription_provider not in ["openai", "elevenlabs", "groq"]:
            raise ValueError("Transcription provider must be 'openai', 'elevenlabs', or 'groq'")
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Create a temporary directory if none provided
            self.temp_dir = tempfile.TemporaryDirectory()
            self.output_dir = self.temp_dir.name
            
        # Initialize transcription module based on provider
        from .transcription import openai_transcriber, elevenlabs_transcriber, groq_transcriber
        
        if transcription_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI transcription")
            self.transcriber = openai_transcriber.OpenAITranscriber(self.openai_api_key, max_workers=max_workers)
        elif transcription_provider == "groq":
            if not self.groq_api_key:
                raise ValueError("Groq API key is required for Groq transcription")
            self.transcriber = groq_transcriber.GroqTranscriber(self.groq_api_key, max_workers=max_workers)
        else:
            if not self.elevenlabs_api_key:
                raise ValueError("ElevenLabs API key is required for ElevenLabs transcription")
            self.transcriber = elevenlabs_transcriber.ElevenLabsTranscriber(self.elevenlabs_api_key)
        
        # Check dependencies
        from .utils import check_dependencies
        if not check_dependencies(self.ffmpeg_path):
            logger.error("Required dependencies are missing. Please install them and try again.")
            sys.exit(1)
    
    def process_youtube_video(
        self,
        youtube_url: str,
        dataset_name: str = None,
        upload_to_hf: bool = False,
        min_segment_seconds: float = 10.0,
        max_segment_seconds: float = 15.0,
        max_minutes: int = None,
        skip_minutes: int = 0,
        chunk_minutes: int = 10
    ) -> Dict:
        """
        Process a YouTube video to create a dataset.
        
        Args:
            youtube_url: URL of the YouTube video
            dataset_name: Name of the dataset (required if upload_to_hf is True)
            upload_to_hf: Whether to upload the dataset to HuggingFace
            min_segment_seconds: Minimum duration of audio segments
            max_segment_seconds: Maximum duration of audio segments
            max_minutes: Maximum minutes of audio to process
            skip_minutes: Minutes to skip from the beginning
            chunk_minutes: Size of chunks for processing long audio
            
        Returns:
            Dictionary with information about the created dataset
        """
        from .utils import download_youtube_audio, convert_to_mp3, create_optimized_segments
        
        logger.info(f"Processing YouTube video: {youtube_url}")
        
        # Create processing directories
        audio_dir = os.path.join(self.output_dir, "audio")
        segments_dir = os.path.join(self.output_dir, "segments")
        dataset_dir = os.path.join(self.output_dir, "dataset")
        
        for dir_path in [audio_dir, segments_dir, dataset_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Step 1: Download YouTube audio
        logger.info("Downloading YouTube audio...")
        audio_file = download_youtube_audio(
            youtube_url, 
            audio_dir,
            max_minutes=max_minutes,
            skip_minutes=skip_minutes
        )
        
        # Step 2: Convert to MP3 if needed
        if not audio_file.endswith('.mp3'):
            mp3_file = os.path.join(audio_dir, "audio.mp3")
            convert_to_mp3(audio_file, mp3_file)
            audio_file = mp3_file
        
        # Step 3: Transcribe audio
        logger.info(f"Transcribing audio using {self.transcription_provider.capitalize()}...")
        transcript = self.transcriber.transcribe_with_timestamps(
            audio_file,
            chunk_minutes=chunk_minutes,
            max_minutes=max_minutes,
            skip_minutes=skip_minutes
        )
        
        # Step 4: Create optimized segments
        logger.info("Creating optimized audio segments...")
        segments = create_optimized_segments(
            audio_file,
            transcript,
            segments_dir,
            min_seconds=min_segment_seconds,
            max_seconds=max_segment_seconds
        )
        
        # Step 5: Create dataset
        from .utils import create_final_dataset
        dataset_info = create_final_dataset(segments, dataset_dir)
        
        # Step 6: Upload to HuggingFace if requested
        if upload_to_hf:
            if not self.hf_token:
                raise ValueError("HuggingFace token is required for uploading")
            
            if not dataset_name:
                raise ValueError("Dataset name is required for HuggingFace upload")
            
            from .utils import upload_to_huggingface
            hf_url = upload_to_huggingface(dataset_info['items'], dataset_name, self.hf_token)
            dataset_info['huggingface_url'] = hf_url
            logger.info(f"Dataset uploaded to HuggingFace: {hf_url}")
        
        return dataset_info

    def process_youtube_videos(
        self,
        youtube_urls: List[str],
        dataset_name: str = None,
        upload_to_hf: bool = False,
        min_segment_seconds: float = 10.0,
        max_segment_seconds: float = 15.0,
        max_minutes: int = None,
        skip_minutes: int = 0,
        chunk_minutes: int = 10
    ) -> List[Dict]:
        """
        Process multiple YouTube videos to create datasets.

        Args:
            youtube_urls: List of YouTube video URLs
            dataset_name: Name of the dataset (required if upload_to_hf is True)
            upload_to_hf: Whether to upload the dataset to HuggingFace
            min_segment_seconds: Minimum duration of audio segments
            max_segment_seconds: Maximum duration of audio segments
            max_minutes: Maximum minutes of audio to process
            skip_minutes: Minutes to skip from the beginning
            chunk_minutes: Size of chunks for processing long audio

        Returns:
            List of dictionaries with information about each created dataset
        """
        results = []

        for i, youtube_url in enumerate(youtube_urls):
            logger.info(f"Processing YouTube video {i+1}/{len(youtube_urls)}: {youtube_url}")

            result = self.process_youtube_video(
                youtube_url=youtube_url,
                dataset_name=dataset_name,
                upload_to_hf=upload_to_hf,
                min_segment_seconds=min_segment_seconds,
                max_segment_seconds=max_segment_seconds,
                max_minutes=max_minutes,
                skip_minutes=skip_minutes,
                chunk_minutes=chunk_minutes
            )

            results.append(result)

        return results

    def __del__(self):
        # Clean up temporary directory if used
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
