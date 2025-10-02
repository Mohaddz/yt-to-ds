"""
Command Line Interface for YTDS.
"""
import os
import sys
import argparse
import logging
from typing import Optional, List

from .ytds import YTDSProcessor

def main(args: Optional[List[str]] = None):
    """
    Run the YTDS CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv if None)
    """
    parser = argparse.ArgumentParser(
        description="YTDS - Convert YouTube videos to transcribed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "youtube_urls",
        nargs='+',
        help="YouTube video URL(s) to process"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to store outputs",
        default=os.path.join(os.getcwd(), "ytds_output")
    )
    
    parser.add_argument(
        "--transcription-provider", "-tp",
        choices=["openai", "elevenlabs", "groq"],
        default="openai",
        help="Transcription provider to use"
    )
    
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--elevenlabs-api-key",
        help="ElevenLabs API key (can also be set via ELEVENLABS_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--groq-api-key",
        help="Groq API key (can also be set via GROQ_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for dataset upload (can also be set via HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--upload-to-hf", "-u",
        action="store_true",
        help="Upload the dataset to HuggingFace"
    )
    
    parser.add_argument(
        "--dataset-name", "-d",
        help="Name of the dataset (required if uploading to HuggingFace)"
    )
    
    parser.add_argument(
        "--ffmpeg-path",
        help="Custom path to ffmpeg binary"
    )
    
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=10.0,
        help="Minimum duration of audio segments in seconds"
    )
    
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=15.0,
        help="Maximum duration of audio segments in seconds"
    )
    
    parser.add_argument(
        "--max-minutes",
        type=int,
        help="Maximum minutes of audio to process"
    )
    
    parser.add_argument(
        "--skip-minutes",
        type=int,
        default=0,
        help="Minutes to skip from the beginning"
    )
    
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=10,
        help="Size of chunks for processing long audio (in minutes)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers for transcription (default: 3)"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate arguments
    if parsed_args.upload_to_hf and not parsed_args.dataset_name:
        parser.error("--dataset-name is required when --upload-to-hf is set")
    
    if parsed_args.transcription_provider == "openai" and not (parsed_args.openai_api_key or os.environ.get("OPENAI_API_KEY")):
        parser.error("OpenAI API key is required for OpenAI transcription (use --openai-api-key or set OPENAI_API_KEY environment variable)")
    
    if parsed_args.transcription_provider == "elevenlabs" and not (parsed_args.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")):
        parser.error("ElevenLabs API key is required for ElevenLabs transcription (use --elevenlabs-api-key or set ELEVENLABS_API_KEY environment variable)")
    
    if parsed_args.transcription_provider == "groq" and not (parsed_args.groq_api_key or os.environ.get("GROQ_API_KEY")):
        parser.error("Groq API key is required for Groq transcription (use --groq-api-key or set GROQ_API_KEY environment variable)")
    
    # Initialize processor
    try:
        processor = YTDSProcessor(
            openai_api_key=parsed_args.openai_api_key,
            elevenlabs_api_key=parsed_args.elevenlabs_api_key,
            groq_api_key=parsed_args.groq_api_key,
            hf_token=parsed_args.hf_token,
            ffmpeg_path=parsed_args.ffmpeg_path,
            output_dir=parsed_args.output_dir,
            transcription_provider=parsed_args.transcription_provider,
            max_workers=parsed_args.max_workers
        )
        
        # Process the YouTube videos
        total_segments = 0
        results = []

        for i, youtube_url in enumerate(parsed_args.youtube_urls):
            print(f"\n🎬 Processing video {i+1}/{len(parsed_args.youtube_urls)}: {youtube_url}")

            result = processor.process_youtube_video(
                youtube_url=youtube_url,
                dataset_name=parsed_args.dataset_name,
                upload_to_hf=parsed_args.upload_to_hf,
                min_segment_seconds=parsed_args.min_segment_seconds,
                max_segment_seconds=parsed_args.max_segment_seconds,
                max_minutes=parsed_args.max_minutes,
                skip_minutes=parsed_args.skip_minutes,
                chunk_minutes=parsed_args.chunk_minutes
            )

            results.append(result)
            total_segments += len(result['items'])

            print(f"✅ Video {i+1} complete! Segments: {len(result['items'])}")
            print(f"   Dataset directory: {os.path.abspath(result['dataset_dir'])}")

            if parsed_args.upload_to_hf and 'huggingface_url' in result:
                print(f"   HuggingFace dataset URL: {result['huggingface_url']}")

        print("\n🎉 All videos processed!")
        print(f"📊 Total segments created across all videos: {total_segments}")

        # Show summary of all results
        for i, result in enumerate(results):
            print(f"   Video {i+1}: {len(result['items'])} segments")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
