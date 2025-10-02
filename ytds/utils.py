"""
Utility functions for YTDS.
"""
import os
import re
import json
import shutil
import logging
import tempfile
import subprocess
import math
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from pydub import AudioSegment
from datasets import Dataset, Features, Audio, Value
from huggingface_hub import HfApi, login
from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_dependencies(ffmpeg_path=None):
    """
    Check if required dependencies are installed.
    
    Args:
        ffmpeg_path: Optional custom path to ffmpeg binary
        
    Returns:
        True if all dependencies are present, False otherwise
    """
    # Check for FFmpeg
    try:
        # If ffmpeg_path is provided, check if it exists
        if ffmpeg_path:
            if os.path.exists(ffmpeg_path):
                logger.info(f"✅ Using FFmpeg from specified path: {ffmpeg_path}")
                # Set the environment variable for subprocesses
                os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
                return True
            else:
                logger.error(f"❌ Specified FFmpeg path does not exist: {ffmpeg_path}")
                return False
                
        # Try to execute ffmpeg to see if it's in the PATH
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode == 0:
            ffmpeg_version = result.stdout.decode('utf-8').split('\n')[0]
            logger.info(f"✅ FFmpeg found: {ffmpeg_version}")
            return True
        else:
            logger.error("FFmpeg check failed with return code: " + str(result.returncode))
            logger.error("Error output: " + result.stderr.decode('utf-8'))
            raise FileNotFoundError("FFmpeg check failed")
            
    except FileNotFoundError:
        logger.error("❌ FFmpeg is not installed or not in your PATH. This is required for audio processing.")
        
        # Try to find ffmpeg in common locations on Windows
        if os.name == 'nt':  # Windows
            common_paths = [
                "C:\\ffmpeg\\bin",
                "C:\\Program Files\\ffmpeg\\bin",
                "C:\\Program Files (x86)\\ffmpeg\\bin"
            ]
            
            for path in common_paths:
                if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                    logger.info(f"✅ Found FFmpeg in {path}, using this location")
                    # Set the environment variable for subprocesses
                    os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                    return True
        
        print("\n===== FFmpeg Installation Instructions =====")
        if os.name == 'nt':  # Windows
            print("Windows:")
            print("1. Download FFmpeg from https://ffmpeg.org/download.html or https://github.com/BtbN/FFmpeg-Builds/releases")
            print("2. Extract the zip to a location like C:\\ffmpeg")
            print("3. Add the bin folder to your PATH: C:\\ffmpeg\\bin")
            print("4. Restart your terminal/command prompt and try again")
            print("5. Alternatively, you can run this script with the --ffmpeg_path option")
        elif os.name == 'posix' and os.uname().sysname == 'Darwin':  # macOS
            print("macOS:")
            print("1. Install with Homebrew: brew install ffmpeg")
            print("2. Or install with MacPorts: port install ffmpeg")
        else:  # Linux
            print("Linux:")
            print("1. Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg")
            print("2. Fedora: sudo dnf install ffmpeg")
            print("3. Arch Linux: sudo pacman -S ffmpeg")
        print("===========================================\n")
        return False


def get_audio_duration(audio_file: str) -> float:
    """
    Get the duration of an audio file in minutes.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Duration in minutes
    """
    try:
        # Try ffprobe first
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        
        if result.returncode == 0 and result.stdout.strip():
            duration_seconds = float(result.stdout.strip())
            if duration_seconds > 0:  # Valid positive duration
                return duration_seconds / 60
    except Exception as e:
        logger.warning(f"Error using ffprobe to get duration: {e}")
    
    # If ffprobe fails, try pydub
    try:
        logger.info("Using pydub fallback to get duration")
        audio = AudioSegment.from_file(audio_file)
        duration_minutes = len(audio) / (60 * 1000)
        return duration_minutes
    except Exception as e:
        logger.error(f"All duration methods failed: {e}")
        
        # As absolute last resort, estimate based on file size
        try:
            size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            # Very rough estimate: ~1MB per minute for mp3 at 192kbps
            estimated_minutes = size_mb * (1.0 if audio_file.endswith('.mp3') else 0.5)
            logger.warning(f"Using size-based estimate: {estimated_minutes:.1f} minutes (VERY UNRELIABLE)")
            return estimated_minutes
        except Exception:
            raise RuntimeError(f"Cannot determine audio duration for {audio_file}")


def safe_copy_file(src, dst):
    """
    Stream-based file copy to ensure no truncation occurs with large files.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst, 1024*1024)  # 1MB buffer


def convert_to_mp3(input_file, output_file):
    """
    Convert audio file to MP3 format.
    
    Args:
        input_file: Path to the input audio file
        output_file: Path to the output MP3 file
    
    Returns:
        Path to the output MP3 file
    """
    if input_file.endswith('.mp3'):
        if input_file != output_file:
            logger.info(f"Input is already MP3, copying to {os.path.basename(output_file)}")
            safe_copy_file(input_file, output_file)
        return output_file
        
    logger.info(f"Converting {os.path.basename(input_file)} to MP3")
    
    try:
        # Normalize paths for Windows
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove output file if it exists (Windows can be finicky with -y flag)
        # Retry removal with delays to handle Windows file locking
        if os.path.exists(output_file):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.remove(output_file)
                    logger.debug(f"Successfully removed existing output file: {output_file}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"File locked, retrying removal in 0.5s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to remove locked file after {max_retries} attempts: {output_file}")
                        raise RuntimeError(f"Cannot remove locked output file: {output_file}. Please close any programs using this file.") from e
                except Exception as e:
                    logger.warning(f"Could not remove existing file: {e}")
                    break
        
        # On Windows, convert backslashes to forward slashes for ffmpeg
        ffmpeg_input = input_file.replace('\\', '/') if os.name == 'nt' else input_file
        ffmpeg_output = output_file.replace('\\', '/') if os.name == 'nt' else output_file
        
        # Use ffmpeg to convert to MP3
        cmd = [
            'ffmpeg',
            '-i', ffmpeg_input, 
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',  # High quality MP3
            '-y',  # Overwrite output file if it exists
            ffmpeg_output
        ]
        
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if process.returncode != 0:
            logger.error(f"Error converting to MP3: {process.stderr.decode('utf-8')}")
            raise RuntimeError(f"Failed to convert {input_file} to MP3")
        
        # Verify the output file was created
        if not os.path.exists(output_file):
            raise RuntimeError(f"Output file was not created: {output_file}")
            
        return output_file
        
    except Exception as e:
        logger.error(f"Exception during MP3 conversion: {e}")
        raise


def download_youtube_audio(video_url: str, output_dir: str, max_minutes: int = None, skip_minutes: int = 0):
    """
    Download YouTube video audio.
    
    Args:
        video_url: YouTube video URL
        output_dir: Directory to store the downloaded audio
        max_minutes: Maximum duration in minutes to download (None for entire video)
        skip_minutes: Minutes to skip from the beginning
        
    Returns:
        Path to the downloaded audio file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old audio files from previous downloads to prevent file locking issues on Windows
    if os.path.exists(output_dir):
        files_to_clean = [f for f in os.listdir(output_dir) if f.endswith(('.m4a', '.mp3', '.webm', '.opus'))]
        if files_to_clean:
            logger.info(f"Cleaning up {len(files_to_clean)} old audio file(s) from previous run...")
        
        for file in files_to_clean:
            file_path = os.path.join(output_dir, file)
            max_retries = 5  # Increased retries for Windows file locking
            for attempt in range(max_retries):
                try:
                    os.remove(file_path)
                    logger.info(f"✓ Removed old audio file: {file}")
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        wait_time = 1.0  # Increased wait time
                        logger.warning(f"⚠ File {file} is locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"✗ Could not remove locked file {file} after {max_retries} attempts")
                        logger.error(f"  Please manually delete: {file_path}")
                        logger.error(f"  Or close any programs that might be using it (Windows Explorer, media players, etc.)")
                        raise RuntimeError(f"Cannot proceed: {file} is locked by another process. Please close all programs using this file and try again.")
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")
                    break
    
    # Progress bar for download
    download_progress = None
    
    def progress_hook(d):
        nonlocal download_progress
        if d['status'] == 'downloading':
            if download_progress is None:
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                download_progress = tqdm(
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading audio"
                )
            downloaded = d.get('downloaded_bytes', 0)
            download_progress.n = downloaded
            download_progress.refresh()
        elif d['status'] == 'finished':
            if download_progress:
                download_progress.close()
                download_progress = None
    
    # Configure yt-dlp options
    # Use video ID in filename to avoid conflicts when multiple videos have similar titles
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s_%(title)s.%(ext)s'),  # Include video ID for uniqueness
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        'restrictfilenames': True,  # Avoid special characters in filenames
        'extract_flat': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',  # Extract as m4a for best quality and time accuracy
            'preferredquality': '192',
        }],
        'logger': logger,
        'progress_hooks': [progress_hook],
    }
    
    # Download the video audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from: {video_url}")
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get('id', 'unknown')
            title = info.get('title', 'video')
            
            # Clean filename - now includes video ID for uniqueness
            clean_title = re.sub(r'[^\w\-_\. ]', '_', title)
            downloaded_file = os.path.join(output_dir, f"{video_id}_{clean_title}.m4a")
            
            if not os.path.exists(downloaded_file):
                # Try to find file by video ID prefix
                potential_files = [f for f in os.listdir(output_dir) if f.startswith(video_id) and f.endswith('.m4a')]
                if potential_files:
                    downloaded_file = os.path.join(output_dir, potential_files[0])
                else:
                    # Fallback: find any m4a file (for backward compatibility)
                    potential_files = [f for f in os.listdir(output_dir) if f.endswith('.m4a')]
                    if potential_files:
                        downloaded_file = os.path.join(output_dir, potential_files[0])
            
            # Apply time limits if specified
            if max_minutes is not None or skip_minutes > 0:
                logger.info(f"Trimming audio: skip={skip_minutes}m, max={max_minutes}m")
                
                # Create a trimmed version
                trimmed_file = os.path.join(output_dir, "trimmed_audio.m4a")
                
                # Convert paths for ffmpeg on Windows
                ffmpeg_input = downloaded_file.replace('\\', '/') if os.name == 'nt' else downloaded_file
                ffmpeg_output = trimmed_file.replace('\\', '/') if os.name == 'nt' else trimmed_file
                
                # Build ffmpeg command for trimming
                cmd = ['ffmpeg', '-i', ffmpeg_input]
                
                # Add seek option if skipping from start
                if skip_minutes > 0:
                    cmd.extend(['-ss', f'{skip_minutes*60}'])
                
                # Add duration limit if max_minutes is specified
                if max_minutes is not None:
                    cmd.extend(['-t', f'{max_minutes*60}'])
                
                # Output file options
                cmd.extend(['-c', 'copy', '-y', ffmpeg_output])
                
                # Run the command
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # Return the trimmed file
                return trimmed_file
            
            return downloaded_file
            
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}")
        raise


def split_audio_into_chunks(
    audio_file: str, 
    chunk_minutes: int = 10, 
    output_dir: str = None,
    max_minutes: int = None,
    skip_minutes: int = 0
) -> List[Dict]:
    """
    Split long audio file into manageable chunks.
    
    Args:
        audio_file: Path to the audio file
        chunk_minutes: Size of each chunk in minutes
        output_dir: Directory to store the chunks (uses temp dir if None)
        max_minutes: Maximum total minutes to process
        skip_minutes: Minutes to skip from the beginning
        
    Returns:
        List of dictionaries with file paths and metadata for each chunk
    """
    # Create temp directory if not provided
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Splitting audio file into {chunk_minutes}-minute chunks")
    
    # Get audio duration
    total_duration_minutes = get_audio_duration(audio_file)
    logger.info(f"Total audio duration: {total_duration_minutes:.2f} minutes")
    
    # Apply limits
    start_minute = skip_minutes
    end_minute = total_duration_minutes
    if max_minutes is not None:
        end_minute = min(end_minute, start_minute + max_minutes)
    
    # Calculate number of chunks
    adjusted_duration = end_minute - start_minute
    num_chunks = math.ceil(adjusted_duration / chunk_minutes)
    
    if num_chunks == 0:
        logger.warning("No audio to process after applying time constraints")
        return []
    
    logger.info(f"Creating {num_chunks} chunks of {chunk_minutes} minutes each")
    
    chunks = []
    with tqdm(total=num_chunks, desc="Splitting audio into chunks", unit="chunk") as pbar:
        for i in range(num_chunks):
            chunk_start = start_minute + (i * chunk_minutes)
            chunk_end = min(chunk_start + chunk_minutes, end_minute)
            
            if chunk_end <= chunk_start:
                break
                
            chunk_duration = chunk_end - chunk_start
            
            # Output file for this chunk
            chunk_file = os.path.join(output_dir, f"chunk_{i+1:03d}.mp3")
            
            # Convert paths for ffmpeg on Windows
            ffmpeg_input = audio_file.replace('\\', '/') if os.name == 'nt' else audio_file
            ffmpeg_output = chunk_file.replace('\\', '/') if os.name == 'nt' else chunk_file
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', ffmpeg_input,
                '-ss', str(chunk_start * 60),  # Start time in seconds
                '-t', str(chunk_duration * 60),  # Duration in seconds
                '-acodec', 'libmp3lame',
                '-q:a', '2',  # High quality
                '-y',  # Overwrite if exists
                ffmpeg_output
            ]
            
            # Execute command
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                chunks.append({
                    'index': i,
                    'file': chunk_file,
                    'start_ms': int(chunk_start * 60 * 1000),  # Start time in ms
                    'duration_ms': int(chunk_duration * 60 * 1000),  # Duration in ms
                })
                
                pbar.update(1)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error creating chunk {i+1}: {e}")
                logger.error(f"FFmpeg error: {e.stderr.decode('utf-8') if e.stderr else 'No error output'}")
                raise
    
    return chunks


def _extract_segment(audio_file: str, segment_data: Dict, output_dir: str, segment_index: int) -> Dict:
    """
    Helper function to extract a single segment using FFmpeg.
    
    Args:
        audio_file: Path to the source audio file
        segment_data: Dictionary with segment metadata
        output_dir: Directory to store the segment
        segment_index: Index number for the segment
        
    Returns:
        Dictionary with segment information or None if extraction failed
    """
    segment_file = os.path.join(output_dir, f"segment_{segment_index:04d}.mp3")
    
    try:
        # Convert paths for ffmpeg on Windows
        ffmpeg_input = audio_file.replace('\\', '/') if os.name == 'nt' else audio_file
        ffmpeg_output = segment_file.replace('\\', '/') if os.name == 'nt' else segment_file
        
        cmd = [
            'ffmpeg',
            '-i', ffmpeg_input,
            '-ss', str(segment_data['start']),
            '-t', str(segment_data['duration']),
            '-acodec', 'libmp3lame',
            '-q:a', '2',
            '-y',
            ffmpeg_output
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        return {
            'audio': segment_file,
            'text': segment_data['text'].strip(),
            'start': segment_data['start'],
            'end': segment_data['end'],
            'duration': segment_data['duration']
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating segment {segment_index}: {e}")
        return None


def create_optimized_segments(
    audio_file: str, 
    transcript_segments: List[Dict], 
    output_dir: str, 
    min_seconds: float = 10.0, 
    max_seconds: float = 15.0,
    max_workers: int = 8
) -> List[Dict]:
    """
    Create audio segments of desired duration from transcript segments with parallel processing.
    
    Args:
        audio_file: Path to the audio file
        transcript_segments: List of transcript segments with start, end times, and text
        output_dir: Directory to store the segments
        min_seconds: Minimum duration of segments in seconds
        max_seconds: Maximum duration of segments in seconds
        max_workers: Maximum number of parallel workers for segment extraction
        
    Returns:
        List of optimized segments with file paths and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating audio segments ({min_seconds}-{max_seconds} seconds each)")
    
    # Sort segments by start time to ensure they're in correct order
    sorted_segments = sorted(transcript_segments, key=lambda x: x['start'])
    
    current_text = ""
    current_start = None
    current_end = None
    current_duration = 0
    
    segments_to_create = []
    
    # Group segments into chunks of desired length
    for segment in sorted_segments:
        segment_duration = segment['end'] - segment['start']
        
        # Skip segments that are too short
        if segment_duration < 0.1:
            continue
            
        # Initialize current chunk if empty
        if current_start is None:
            current_start = segment['start']
            current_text = segment['text']
            current_end = segment['end']
            current_duration = segment_duration
        # Add to current chunk if it won't exceed max duration
        elif current_duration + segment_duration <= max_seconds:
            current_text += " " + segment['text']
            current_end = segment['end']
            current_duration += segment_duration
        # Current chunk is full, process it if it meets min duration
        else:
            if current_duration >= min_seconds:
                segments_to_create.append({
                    'text': current_text,
                    'start': current_start,
                    'end': current_end,
                    'duration': current_duration
                })
            
            # Start a new chunk with current segment
            current_start = segment['start']
            current_text = segment['text']
            current_end = segment['end']
            current_duration = segment_duration
    
    # Process the last chunk if it exists and meets minimum duration
    if current_start is not None and current_duration >= min_seconds:
        segments_to_create.append({
            'text': current_text,
            'start': current_start,
            'end': current_end,
            'duration': current_duration
        })
    
    # Extract segments in parallel
    dataset_items = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(_extract_segment, audio_file, seg_data, output_dir, idx + 1): idx
            for idx, seg_data in enumerate(segments_to_create)
        }
        
        with tqdm(total=len(segments_to_create), desc="Creating audio segments", unit="segment") as pbar:
            for future in as_completed(future_to_segment):
                result = future.result()
                if result:
                    dataset_items.append(result)
                pbar.update(1)
    
    # Sort by start time to maintain order
    dataset_items.sort(key=lambda x: x['start'])
    
    logger.info(f"Created {len(dataset_items)} optimized segments")
    return dataset_items


def create_final_dataset(segments: List[Dict], output_dir: str) -> Dict:
    """
    Create final dataset with audio segments.
    
    Args:
        segments: List of audio segments with metadata
        output_dir: Directory to store the dataset
        
    Returns:
        Dictionary with dataset information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating dataset with {len(segments)} segments")
    
    # Create dataset directory structure
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Copy audio files to dataset directory and create metadata
    dataset_items = []
    for i, segment in enumerate(segments):
        # Output file path
        output_file = os.path.join(audio_dir, f"segment_{i+1:04d}.mp3")
        
        # Copy audio file
        safe_copy_file(segment['audio'], output_file)
        
        # Update dataset item
        dataset_items.append({
            'audio': output_file,
            'text': segment['text'],
            'start_time': segment['start'],
            'end_time': segment['end'],
            'duration': segment['duration']
        })
    
    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_items, f, ensure_ascii=False, indent=2)
    
    # Create Hugging Face dataset
    hf_dataset = Dataset.from_dict({
        'audio': [item['audio'] for item in dataset_items],
        'text': [item['text'] for item in dataset_items],
        'start_time': [item['start_time'] for item in dataset_items],
        'end_time': [item['end_time'] for item in dataset_items],
        'duration': [item['duration'] for item in dataset_items]
    })
    
    # Cast audio column to Audio feature
    hf_dataset = hf_dataset.cast_column('audio', Audio())
    
    # Save dataset locally
    hf_dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    
    return {
        'items': dataset_items,
        'dataset_dir': output_dir,
        'metadata_file': metadata_file
    }


def create_dataset_readme(dataset_items: List[Dict], dataset_name: str, video_count: int = 1) -> str:
    """
    Create a README.md content for the HuggingFace dataset.
    
    Args:
        dataset_items: List of dataset items
        dataset_name: Name of the dataset on HuggingFace
        video_count: Number of source videos
        
    Returns:
        README.md content as string
    """
    # Calculate statistics
    total_duration_seconds = sum(item['duration'] for item in dataset_items)
    total_hours = total_duration_seconds / 3600
    total_minutes = total_duration_seconds / 60
    num_segments = len(dataset_items)
    
    avg_segment_duration = total_duration_seconds / num_segments if num_segments > 0 else 0
    
    # Detect language from text samples (simple heuristic)
    sample_texts = ' '.join([item['text'][:100] for item in dataset_items[:5]])
    language = "Unknown"
    if any(ord(char) > 0x0600 and ord(char) < 0x06FF for char in sample_texts):
        language = "Arabic (ar)"
    elif all(ord(char) < 128 for char in sample_texts.replace(' ', '')):
        language = "English (en)"
    
    readme_content = f"""---
language:
- {language.split('(')[-1].strip(')') if '(' in language else 'unknown'}
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
pretty_name: Audio Transcription Dataset
size_categories:
- {_get_size_category(num_segments)}
---

# Audio Transcription Dataset

This dataset was automatically generated using [yt-to-ds](https://github.com/yourusername/yt-to-ds) for speech recognition and audio transcription tasks.

## Dataset Statistics

- **Total Audio Duration**: {total_hours:.2f} hours ({total_minutes:.1f} minutes)
- **Number of Segments**: {num_segments:,}
- **Source Videos**: {video_count}
- **Average Segment Length**: {avg_segment_duration:.1f} seconds
- **Language**: {language}

## Dataset Structure

Each example in the dataset contains:

- `audio`: Audio file (MP3 format)
- `text`: Transcribed text
- `start_time`: Start time in the original audio (seconds)
- `end_time`: End time in the original audio (seconds)
- `duration`: Duration of the segment (seconds)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}")

# Access an example
example = dataset["train"][0]
print(example["text"])
# Play audio
from IPython.display import Audio
Audio(example["audio"]["array"], rate=example["audio"]["sampling_rate"])
```

## Dataset Creation

This dataset was created by:
1. Downloading audio from YouTube videos
2. Transcribing using Whisper API (OpenAI/Groq)
3. Segmenting into {avg_segment_duration:.0f}-second chunks
4. Aligning transcriptions with audio segments

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{audio_transcription_dataset,
  title={{Audio Transcription Dataset}},
  author={{Generated using yt-to-ds}},
  year={{2025}},
  publisher={{Hugging Face}},
}}
```

## License

This dataset is released under the CC-BY-4.0 license.
"""
    
    return readme_content


def _get_size_category(num_segments: int) -> str:
    """Get HuggingFace size category based on number of segments."""
    if num_segments < 100:
        return "n<1K"
    elif num_segments < 1000:
        return "1K<n<10K"
    elif num_segments < 10000:
        return "10K<n<100K"
    elif num_segments < 100000:
        return "100K<n<1M"
    else:
        return "n>1M"


def upload_to_huggingface(
    dataset_items: List[Dict], 
    dataset_name: str, 
    hf_token: str,
    video_count: int = 1
) -> str:
    """
    Upload dataset to Hugging Face with auto-generated README.
    
    Args:
        dataset_items: List of dataset items
        dataset_name: Name of the dataset on Hugging Face
        hf_token: Hugging Face token
        video_count: Number of source videos
        
    Returns:
        URL of the uploaded dataset
    """
    logger.info(f"Uploading dataset to Hugging Face: {dataset_name}")
    
    # Prepare dataset for upload
    hf_dataset = Dataset.from_dict({
        'audio': [item['audio'] for item in dataset_items],
        'text': [item['text'] for item in dataset_items],
        'start_time': [item['start_time'] for item in dataset_items],
        'end_time': [item['end_time'] for item in dataset_items],
        'duration': [item['duration'] for item in dataset_items]
    })
    
    # Cast audio column to Audio feature
    hf_dataset = hf_dataset.cast_column('audio', Audio())
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Create README content
    readme_content = create_dataset_readme(dataset_items, dataset_name, video_count)
    
    # Push to Hugging Face with README
    repo_url = hf_dataset.push_to_hub(dataset_name, token=hf_token)
    
    # Upload README separately using HfApi
    try:
        api = HfApi()
        
        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(readme_content)
            readme_path = f.name
        
        # Upload README to the dataset repo
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset",
            token=hf_token
        )
        
        # Clean up temporary file
        os.unlink(readme_path)
        
        logger.info("✅ README.md uploaded to HuggingFace dataset")
    except Exception as e:
        logger.warning(f"Could not upload README to HuggingFace: {e}")
    
    return repo_url
