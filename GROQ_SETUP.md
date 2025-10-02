# Groq Whisper Setup Guide

## What is Groq?

Groq provides ultra-fast AI inference, including Whisper models for speech-to-text transcription. Their API is OpenAI-compatible, making it a drop-in replacement that's both faster and more cost-effective.

## Why Use Groq?

1. **Speed**: 50-70% faster API responses compared to OpenAI
2. **Cost**: Very competitive pricing with generous free tier
3. **Quality**: Same Whisper models (whisper-large-v3)
4. **Reliability**: High uptime and rate limits

## Getting Started

### 1. Get a Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy your API key (starts with `gsk_...`)

### 2. Set Up Your Environment

#### Option A: Environment Variable (Recommended)
```bash
# Linux/macOS
export GROQ_API_KEY="gsk_your_api_key_here"

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_api_key_here"

# Windows Command Prompt
set GROQ_API_KEY=gsk_your_api_key_here
```

#### Option B: Command Line Argument
```bash
ytds VIDEO_URL --transcription-provider groq --groq-api-key gsk_your_api_key_here
```

### 3. Use with YTDS

#### Basic Usage
```bash
ytds https://www.youtube.com/watch?v=example --transcription-provider groq
```

#### Optimized for Speed
```bash
ytds https://www.youtube.com/watch?v=example \
  --transcription-provider groq \
  --max-workers 10
```

#### Python API
```python
from ytds import YTDSProcessor

processor = YTDSProcessor(
    groq_api_key="gsk_your_api_key_here",
    transcription_provider="groq",
    max_workers=10  # Groq handles high concurrency well
)

result = processor.process_youtube_video(
    youtube_url="https://www.youtube.com/watch?v=example",
    min_segment_seconds=10.0,
    max_segment_seconds=15.0
)
```

## Performance Tips

### Optimal Worker Count
- **Light usage**: `--max-workers 3-5`
- **Normal usage**: `--max-workers 5-7`
- **Maximum speed**: `--max-workers 10-15`

### Rate Limits
Groq has generous rate limits. The free tier typically allows:
- Multiple concurrent requests
- High tokens per minute
- Suitable for processing multiple videos

### Processing Long Videos
For videos longer than 60 minutes:
```bash
ytds VIDEO_URL \
  --transcription-provider groq \
  --max-workers 10 \
  --chunk-minutes 10
```

## Troubleshooting

### "Invalid API Key" Error
- Verify your API key is correct
- Check that it starts with `gsk_`
- Ensure no extra spaces or quotes in the key

### Rate Limit Errors
- Reduce `--max-workers` to 3-5
- Wait a few minutes before retrying
- Check your Groq dashboard for rate limit details

### Slower Than Expected
- Increase `--max-workers` (try 10)
- Check your internet connection
- Verify you're using the `groq` provider, not `openai`

## Model Information

YTDS uses Groq's `whisper-large-v3` model by default, which provides:
- Excellent accuracy across multiple languages
- Fast inference times
- Support for timestamp granularity

## Cost Comparison (Approximate)

For a 60-minute video:

| Provider | Cost | Time | Quality |
|----------|------|------|---------|
| Groq | Free / Very Low | ~2-3 min | Excellent |
| OpenAI | ~$0.36 | ~4-6 min | Excellent |
| ElevenLabs | Varies | ~5-7 min | Good |

## Example Workflows

### Quick Test (1-2 minute video)
```bash
ytds "https://www.youtube.com/watch?v=short_video" \
  --transcription-provider groq \
  --max-minutes 2
```

### Full Production Run
```bash
ytds "https://www.youtube.com/watch?v=full_video" \
  --transcription-provider groq \
  --max-workers 10 \
  --upload-to-hf \
  --dataset-name username/my-dataset \
  --output-dir ./my_dataset
```

### Batch Processing
```bash
ytds \
  "https://www.youtube.com/watch?v=video1" \
  "https://www.youtube.com/watch?v=video2" \
  "https://www.youtube.com/watch?v=video3" \
  --transcription-provider groq \
  --max-workers 10
```

## Support

For Groq-specific issues:
- Visit [Groq Documentation](https://console.groq.com/docs)
- Check [Groq Discord](https://discord.gg/groq)
- Review rate limits in your [Groq Console](https://console.groq.com)

For YTDS issues:
- Check the main README.md
- Review PERFORMANCE.md for optimization tips

