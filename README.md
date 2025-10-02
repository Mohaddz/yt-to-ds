# YTDS - YouTube to Dataset

A Python tool for converting YouTube videos into transcribed audio datasets. This tool downloads audio from YouTube videos, transcribes them, and creates an optimized dataset that can be used for machine learning or other purposes.

## Features

- Download audio from YouTube videos
- Support for multiple YouTube videos in a single run
- Transcription using OpenAI Whisper API, Groq Whisper API, or ElevenLabs Speech-to-Text API
- **Parallel processing** for faster transcription and segment creation
- Segment audio into optimal chunks based on transcript content
- Create a structured dataset from audio segments
- Optional upload to Hugging Face Hub
- Both Python API and Command Line Interface

## Installation

Install from source:
```bash
git clone https://github.com/yourusername/ytds.git
cd ytds
pip install -e .
```

## Requirements

- Python 3.7+
- FFmpeg (required for audio processing)
- OpenAI API key (for OpenAI Whisper transcription)
- Groq API key (optional, for Groq Whisper transcription - **faster and cheaper**)
- ElevenLabs API key (optional, for ElevenLabs transcription)
- HuggingFace token (optional, for dataset upload)

## Usage

### Python API

```python
from ytds import YTDSProcessor

# Initialize with OpenAI for transcription
processor = YTDSProcessor(
    openai_api_key="your_openai_api_key",
    hf_token="your_huggingface_token",  # Optional
    output_dir="./output",  # Optional
    transcription_provider="openai",  # Default
    max_workers=3  # Parallel workers for faster processing
)

# Or initialize with Groq for transcription (faster and cheaper!)
processor = YTDSProcessor(
    groq_api_key="your_groq_api_key",
    hf_token="your_huggingface_token",  # Optional
    output_dir="./output",  # Optional
    transcription_provider="groq",
    max_workers=5  # Groq can handle more parallel requests
)

# Or initialize with ElevenLabs for transcription
processor = YTDSProcessor(
    elevenlabs_api_key="your_elevenlabs_api_key",
    hf_token="your_huggingface_token",  # Optional
    output_dir="./output",  # Optional
    transcription_provider="elevenlabs"
)

# Process a single YouTube video
result = processor.process_youtube_video(
    youtube_url="https://www.youtube.com/watch?v=example",
    dataset_name="my-youtube-dataset",  # Required for HuggingFace upload
    upload_to_hf=True,  # Optional, defaults to False
    min_segment_seconds=10.0,  # Optional
    max_segment_seconds=15.0,  # Optional
    max_minutes=None,  # Optional, limit processing to first N minutes
    skip_minutes=0,  # Optional, skip first N minutes
    chunk_minutes=10  # Optional, chunk size for processing long audio
)

# Process multiple YouTube videos at once
video_urls = [
    "https://www.youtube.com/watch?v=example1",
    "https://www.youtube.com/watch?v=example2",
    "https://www.youtube.com/watch?v=example3"
]

results = processor.process_youtube_videos(
    youtube_urls=video_urls,
    dataset_name="my-youtube-dataset",
    upload_to_hf=True,
    min_segment_seconds=10.0,
    max_segment_seconds=15.0
)

# Access the results
print(f"Single video: {len(result['items'])} segments")
print(f"Dataset directory: {result['dataset_dir']}")

for i, result in enumerate(results):
    print(f"Video {i+1}: {len(result['items'])} segments")
    print(f"Dataset directory: {result['dataset_dir']}")
    if 'huggingface_url' in result:
        print(f"HuggingFace dataset URL: {result['huggingface_url']}")
```

### Command Line Interface

```bash
# Basic usage with OpenAI (single video)
ytds https://www.youtube.com/watch?v=example --openai-api-key YOUR_API_KEY

# Use Groq for faster and cheaper transcription
ytds https://www.youtube.com/watch?v=example --transcription-provider groq --groq-api-key YOUR_API_KEY

# Process multiple videos at once with parallel processing
ytds https://www.youtube.com/watch?v=example1 https://www.youtube.com/watch?v=example2 --groq-api-key YOUR_API_KEY --transcription-provider groq --max-workers 5

# Use ElevenLabs instead
ytds https://www.youtube.com/watch?v=example --transcription-provider elevenlabs --elevenlabs-api-key YOUR_API_KEY

# Upload to HuggingFace
ytds https://www.youtube.com/watch?v=example --openai-api-key YOUR_API_KEY --upload-to-hf --dataset-name my-youtube-dataset --hf-token YOUR_HF_TOKEN

# Customize segment lengths (in seconds)
ytds https://www.youtube.com/watch?v=example --min-segment-seconds 5 --max-segment-seconds 10

# Process only a portion of the video
ytds https://www.youtube.com/watch?v=example --max-minutes 30 --skip-minutes 5

# Specify output directory
ytds https://www.youtube.com/watch?v=example --output-dir ./my_dataset

# For more options
ytds --help
```

## Environment Variables

You can use environment variables instead of passing API keys directly:

```bash
# Set environment variables
export OPENAI_API_KEY=your_openai_api_key
export GROQ_API_KEY=your_groq_api_key
export ELEVENLABS_API_KEY=your_elevenlabs_api_key
export HF_TOKEN=your_huggingface_token

# Then run without explicitly providing the keys
ytds https://www.youtube.com/watch?v=example --transcription-provider groq
```

## Performance Optimizations

This tool includes several performance optimizations:

1. **Parallel Chunk Transcription**: Audio chunks are transcribed simultaneously using ThreadPoolExecutor
2. **Parallel Segment Extraction**: Audio segments are extracted in parallel using FFmpeg
3. **Configurable Workers**: Control the level of parallelism with `--max-workers` (default: 3)

### Performance Tips

- **Groq**: Supports higher parallelism (recommended: `--max-workers 5-10`) and is significantly faster and cheaper than OpenAI
- **OpenAI**: Use `--max-workers 3-5` to balance speed and API rate limits
- **ElevenLabs**: Sequential processing to avoid rate limiting

Example for maximum speed with Groq:
```bash
ytds https://www.youtube.com/watch?v=example --transcription-provider groq --groq-api-key YOUR_KEY --max-workers 10
```

## License

This project is created by Mohad and is available under the MIT License.
