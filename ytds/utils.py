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
from typing import List, Dict, Tuple, Optional

import yt_dlp
from pydub import AudioSegment
from datasets import Dataset, Features, Audio, Value
from huggingface_hub import HfApi, login

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
        # Use ffmpeg to convert to MP3
        cmd = [
            'ffmpeg', '-i', input_file, 
            '-codec:a', 'libmp3lame', '-qscale:a', '2',  # High quality MP3
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if process.returncode != 0:
            logger.error(f"Error converting to MP3: {process.stderr.decode('utf-8')}")
            raise RuntimeError(f"Failed to convert {input_file} to MP3")
            
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
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
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
    }
    
    # Download the video audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from: {video_url}")
            info = ydl.extract_info(video_url, download=True)
            title = info.get('title', 'video')
            
            # Clean filename
            clean_title = re.sub(r'[^\w\-_\. ]', '_', title)
            downloaded_file = os.path.join(output_dir, f"{clean_title}.m4a")
            
            if not os.path.exists(downloaded_file):
                # Try alternative filename
                potential_files = [f for f in os.listdir(output_dir) if f.endswith('.m4a')]
                if potential_files:
                    downloaded_file = os.path.join(output_dir, potential_files[0])
            
            # Apply time limits if specified
            if max_minutes is not None or skip_minutes > 0:
                logger.info(f"Trimming audio: skip={skip_minutes}m, max={max_minutes}m")
                
                # Create a trimmed version
                trimmed_file = os.path.join(output_dir, "trimmed_audio.m4a")
                
                # Build ffmpeg command for trimming
                cmd = ['ffmpeg', '-i', downloaded_file]
                
                # Add seek option if skipping from start
                if skip_minutes > 0:
                    cmd.extend(['-ss', f'{skip_minutes*60}'])
                
                # Add duration limit if max_minutes is specified
                if max_minutes is not None:
                    cmd.extend(['-t', f'{max_minutes*60}'])
                
                # Output file options
                cmd.extend(['-c', 'copy', '-y', trimmed_file])
                
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
    for i in range(num_chunks):
        chunk_start = start_minute + (i * chunk_minutes)
        chunk_end = min(chunk_start + chunk_minutes, end_minute)
        
        if chunk_end <= chunk_start:
            break
            
        chunk_duration = chunk_end - chunk_start
        
        # Output file for this chunk
        chunk_file = os.path.join(output_dir, f"chunk_{i+1:03d}.mp3")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', audio_file,
            '-ss', str(chunk_start * 60),  # Start time in seconds
            '-t', str(chunk_duration * 60),  # Duration in seconds
            '-acodec', 'libmp3lame',
            '-q:a', '2',  # High quality
            '-y',  # Overwrite if exists
            chunk_file
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
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating chunk {i+1}: {e}")
            logger.error(f"FFmpeg error: {e.stderr.decode('utf-8') if e.stderr else 'No error output'}")
            raise
    
    return chunks


def create_optimized_segments(
    audio_file: str, 
    transcript_segments: List[Dict], 
    output_dir: str, 
    min_seconds: float = 10.0, 
    max_seconds: float = 15.0
) -> List[Dict]:
    """
    Create audio segments of desired duration from transcript segments.
    
    Args:
        audio_file: Path to the audio file
        transcript_segments: List of transcript segments with start, end times, and text
        output_dir: Directory to store the segments
        min_seconds: Minimum duration of segments in seconds
        max_seconds: Maximum duration of segments in seconds
        
    Returns:
        List of optimized segments with file paths and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating audio segments ({min_seconds}-{max_seconds} seconds each)")
    
    # Sort segments by start time to ensure they're in correct order
    sorted_segments = sorted(transcript_segments, key=lambda x: x['start'])
    
    current_segments = []
    current_text = ""
    current_start = None
    current_end = None
    current_duration = 0
    
    dataset_items = []
    segment_index = 0
    
    # Group segments into chunks of desired length
    for segment in sorted_segments:
        segment_duration = segment['end'] - segment['start']
        
        # Skip segments that are too short (less than 0.1 seconds)
        if segment_duration < 0.1:
            continue
            
        # Initialize current chunk if empty
        if current_start is None:
            current_start = segment['start']
            current_text = segment['text']
            current_end = segment['end']
            current_duration = segment_duration
            current_segments = [segment]
        # Add to current chunk if it won't exceed max duration
        elif current_duration + segment_duration <= max_seconds:
            current_text += " " + segment['text']
            current_end = segment['end']
            current_duration += segment_duration
            current_segments.append(segment)
        # Current chunk is full, process it if it meets min duration
        else:
            # If current chunk meets minimum duration, process it
            if current_duration >= min_seconds:
                segment_index += 1
                segment_file = os.path.join(output_dir, f"segment_{segment_index:04d}.mp3")
                
                # Extract segment from audio file using ffmpeg
                try:
                    cmd = [
                        'ffmpeg',
                        '-i', audio_file,
                        '-ss', str(current_start),  # Start time in seconds
                        '-t', str(current_duration),  # Duration in seconds
                        '-acodec', 'libmp3lame',
                        '-q:a', '2',  # High quality
                        '-y',  # Overwrite if exists
                        segment_file
                    ]
                    
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    
                    # Add to dataset items
                    dataset_items.append({
                        'audio': segment_file,
                        'text': current_text.strip(),
                        'start': current_start,
                        'end': current_end,
                        'duration': current_duration
                    })
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error creating segment {segment_index}: {e}")
            
            # Start a new chunk with current segment
            current_start = segment['start']
            current_text = segment['text']
            current_end = segment['end']
            current_duration = segment_duration
            current_segments = [segment]
    
    # Process the last chunk if it exists and meets minimum duration
    if current_start is not None and current_duration >= min_seconds:
        segment_index += 1
        segment_file = os.path.join(output_dir, f"segment_{segment_index:04d}.mp3")
        
        # Extract segment from audio file
        try:
            cmd = [
                'ffmpeg',
                '-i', audio_file,
                '-ss', str(current_start),  # Start time in seconds
                '-t', str(current_duration),  # Duration in seconds
                '-acodec', 'libmp3lame',
                '-q:a', '2',  # High quality
                '-y',  # Overwrite if exists
                segment_file
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Add to dataset items
            dataset_items.append({
                'audio': segment_file,
                'text': current_text.strip(),
                'start': current_start,
                'end': current_end,
                'duration': current_duration
            })
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating segment {segment_index}: {e}")
    
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


def upload_to_huggingface(dataset_items: List[Dict], dataset_name: str, hf_token: str) -> str:
    """
    Upload dataset to Hugging Face.
    
    Args:
        dataset_items: List of dataset items
        dataset_name: Name of the dataset on Hugging Face
        hf_token: Hugging Face token
        
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
    
    # Push to Hugging Face
    repo_url = hf_dataset.push_to_hub(dataset_name, token=hf_token)
    
    return repo_url
