import os
import re
import json
import logging
import argparse
import tempfile
import sys
import subprocess
import math
from typing import List, Dict, Tuple
import shutil

import yt_dlp
import openai
from pydub import AudioSegment
from datasets import Dataset, Features, Audio, Value
from huggingface_hub import HfApi, login

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies(ffmpeg_path=None):
    """Check if required dependencies are installed"""
    # Check for FFmpeg
    try:
        # If ffmpeg_path is provided, check if it exists
        if ffmpeg_path:
            if os.path.exists(ffmpeg_path):
                print(f"✅ Using FFmpeg from specified path: {ffmpeg_path}")
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
            print(f"✅ FFmpeg found: {ffmpeg_version}")
            return True
        else:
            logger.error("FFmpeg check failed with return code: " + str(result.returncode))
            logger.error("Error output: " + result.stderr.decode('utf-8'))
            raise FileNotFoundError("FFmpeg check failed")
            
    except FileNotFoundError:
        logger.error("FFmpeg is not installed or not in your PATH. This is required for audio processing.")
        
        # Try to find ffmpeg in common locations on Windows
        if sys.platform.startswith('win'):
            common_paths = [
                "C:\\ffmpeg\\bin",
                "C:\\Program Files\\ffmpeg\\bin",
                "C:\\Program Files (x86)\\ffmpeg\\bin"
            ]
            
            for path in common_paths:
                if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                    print(f"✅ Found FFmpeg in {path}, using this location")
                    # Set the environment variable for subprocesses
                    os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                    return True
        
        print("\n===== FFmpeg Installation Instructions =====")
        if sys.platform.startswith('win'):
            print("Windows:")
            print("1. Download FFmpeg from https://ffmpeg.org/download.html or https://github.com/BtbN/FFmpeg-Builds/releases")
            print("2. Extract the zip to a location like C:\\ffmpeg")
            print("3. Add the bin folder to your PATH: C:\\ffmpeg\\bin")
            print("4. Restart your terminal/command prompt and try again")
            print("5. Alternatively, you can run this script with the --ffmpeg_path option:")
            print("   python script.py --ffmpeg_path C:\\path\\to\\ffmpeg\\bin")
        elif sys.platform.startswith('darwin'):
            print("macOS:")
            print("1. Install with Homebrew: brew install ffmpeg")
            print("2. Or install with MacPorts: port install ffmpeg")
        else:
            print("Linux:")
            print("1. Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg")
            print("2. Fedora: sudo dnf install ffmpeg")
            print("3. Arch Linux: sudo pacman -S ffmpeg")
        print("===========================================\n")
        return False

def verify_api_keys(openai_api_key: str, hf_token: str):
    """Verify that the API keys are valid"""
    # Verify OpenAI API key
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        client.models.list()
        logger.info("✅ OpenAI API key verified successfully")
    except Exception as e:
        logger.error(f"❌ OpenAI API key verification failed: {e}")
        return False
    
    # Verify HuggingFace token
    try:
        api = HfApi(token=hf_token)
        api.whoami()
        logger.info("✅ Hugging Face token verified successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Hugging Face token verification failed: {e}")
        return False

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file in minutes with enhanced error handling."""
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        attempts += 1
        try:
            # Try ffprobe first
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                duration_seconds = float(result.stdout.strip())
                if duration_seconds > 0:  # Valid positive duration
                    return duration_seconds / 60
                else:
                    print(f"⚠️ Warning: ffprobe returned duration of {duration_seconds} seconds")
            else:
                print(f"⚠️ ffprobe failed (attempt {attempts}/{max_attempts})")
                print(f"   Error: {result.stderr}")
                
            # If we're here, ffprobe failed - try a different approach if not last attempt
            if attempts < max_attempts:
                print(f"   Trying alternative method...")
                time.sleep(1)  # Brief pause before retry
            
        except Exception as e:
            print(f"⚠️ Error with ffprobe (attempt {attempts}/{max_attempts}): {e}")
            
    # If all ffprobe attempts failed, try pydub as fallback
    try:
        print(f"🔄 Using pydub fallback to get duration...")
        audio = AudioSegment.from_file(audio_file)
        duration_minutes = len(audio) / (60 * 1000)
        print(f"   - Got duration via pydub: {duration_minutes:.1f} minutes")
        return duration_minutes
    except Exception as e2:
        print(f"❌ All duration methods failed: {e2}")
        # As absolute last resort, estimate based on file size
        try:
            size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            # Very rough estimate: ~1MB per minute for mp3 at 192kbps
            estimated_minutes = size_mb * (1.0 if audio_file.endswith('.mp3') else 0.5)
            print(f"⚠️ Using size-based estimate: {estimated_minutes:.1f} minutes (VERY UNRELIABLE)")
            return estimated_minutes
        except Exception:
            raise RuntimeError(f"Cannot determine audio duration for {audio_file}")

def create_optimized_segments(audio_file: str, transcript_segments: List[Dict], output_dir: str, 
                             min_seconds: float = 10.0, max_seconds: float = 15.0) -> List[Dict]:
    """Create audio segments of desired duration from transcript"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"⏱️ Creating audio segments ({min_seconds}-{max_seconds} seconds each)...")
    
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
            if current_duration >= min_seconds:
                # Extract this segment directly from the source audio
                segment_file = os.path.join(output_dir, f"segment_{segment_index:04d}.mp3")
                
                # Use ffmpeg to extract the segment
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_file,
                    '-ss', str(current_start),
                    '-to', str(current_end),
                    '-c:a', 'libmp3lame',
                    '-b:a', '192k',
                    segment_file
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Add to dataset
                    dataset_items.append({
                        "text": current_text.strip(),
                        "audio": segment_file,
                        "start": current_start,
                        "end": current_end,
                        "duration": current_duration,
                        "letter_count": len(current_text)
                    })
                    
                    segment_index += 1
                    
                    # Print progress periodically
                    if segment_index % 50 == 0:
                        print(f"   - Created {segment_index} segments so far...")
                        
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error creating segment {segment_index}: {e}")
            
            # Start a new chunk with the current segment
            current_start = segment['start']
            current_text = segment['text']
            current_end = segment['end']
            current_duration = segment_duration
            current_segments = [segment]
    
    # Process the last chunk if it meets criteria
    if current_start is not None and current_duration >= min_seconds:
        segment_file = os.path.join(output_dir, f"segment_{segment_index:04d}.mp3")
        
        # Use ffmpeg to extract the segment
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-ss', str(current_start),
            '-to', str(current_end),
            '-c:a', 'libmp3lame',
            '-b:a', '192k',
            segment_file
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Add to dataset
            dataset_items.append({
                "text": current_text.strip(),
                "audio": segment_file,
                "start": current_start,
                "end": current_end,
                "duration": current_duration,
                "letter_count": len(current_text)
            })
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating final segment: {e}")
    
    print(f"✅ Created {len(dataset_items)} audio segments")
    
    # Print statistics
    durations = [item["duration"] for item in dataset_items]
    letter_counts = [item["letter_count"] for item in dataset_items]
    
    if dataset_items:
        print(f"📊 Segment Statistics:")
        print(f"   - Duration: min={min(durations):.1f}s, avg={sum(durations)/len(durations):.1f}s, max={max(durations):.1f}s")
        print(f"   - Letter count: min={min(letter_counts)}, avg={sum(letter_counts)/len(letter_counts):.1f}, max={max(letter_counts)}")
    
    return dataset_items

def estimate_whisper_cost(audio_duration_minutes: float) -> float:
    """Estimate the approximate cost of transcribing audio with OpenAI's Whisper API"""
    # As of March 2025, Whisper-1 costs approximately $0.006 per minute
    rate_per_minute = 0.006
    return round(audio_duration_minutes * rate_per_minute, 2)

def safe_copy_file(src, dst):
    """Stream-based file copy to ensure no truncation occurs with large files"""
    print(f"🔄 Safely copying {os.path.basename(src)} to {os.path.basename(dst)}...")
    
    src_size = os.path.getsize(src)
    try:
        with open(src, 'rb') as src_file:
            with open(dst, 'wb') as dst_file:
                chunk_size = 1024 * 1024  # 1MB chunks
                bytes_copied = 0
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    dst_file.write(chunk)
                    bytes_copied += len(chunk)
                    # Progress update for large files
                    if src_size > 100 * 1024 * 1024:  # Over 100MB
                        print(f"   - Progress: {bytes_copied / src_size * 100:.1f}%", end="\r")
        
        # Verify file sizes match
        dst_size = os.path.getsize(dst)
        if src_size != dst_size:
            print(f"⚠️ WARNING: File size mismatch after copy!")
            print(f"   - Source: {src_size / (1024 * 1024):.1f} MB")
            print(f"   - Destination: {dst_size / (1024 * 1024):.1f} MB")
            raise IOError(f"File copy failed - size mismatch")
            
        print(f"✅ File copied successfully: {src_size / (1024 * 1024):.1f} MB")
        return dst
    except Exception as e:
        print(f"❌ Error during file copy: {e}")
        if os.path.exists(dst):
            os.remove(dst)  # Clean up partial file
        raise

def convert_to_mp3(input_file, output_file):
    """Convert audio file to MP3 format with careful error handling"""
    print(f"🔄 Converting {os.path.basename(input_file)} to MP3...")
    
    try:
        # First check duration of input file
        input_duration = get_audio_duration(input_file)
        print(f"   - Input file duration: {input_duration:.1f} minutes")
        
        # Convert using explicit settings
        cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
            '-acodec', 'libmp3lame',
            '-b:a', '192k',
            output_file
        ]
        
        # Run the command
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Verify command success
        if process.returncode != 0:
            print(f"⚠️ FFmpeg conversion warning - return code: {process.returncode}")
            print(f"FFmpeg stderr: {process.stderr[:500]}...")  # Print first 500 chars of stderr
        
        # Verify output duration
        if os.path.exists(output_file):
            output_duration = get_audio_duration(output_file)
            print(f"   - Output file duration: {output_duration:.1f} minutes")
            
            if abs(output_duration - input_duration) > 1.0:  # Allow 1 minute difference
                print(f"⚠️ WARNING: Duration mismatch!")
                print(f"   - Input: {input_duration:.1f} minutes")
                print(f"   - Output: {output_duration:.1f} minutes")
                return False
            
            print(f"✅ Conversion successful - Duration: {output_duration:.1f} minutes")
            return True
        else:
            print(f"❌ Conversion failed - Output file doesn't exist")
            return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False
    


def download_youtube_audio(video_url: str, output_dir: str, max_minutes: int = None, skip_minutes: int = 0) -> str:
    """Download YouTube video audio without processing it into MP3 yet."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"⏱️ Downloading audio from YouTube... This may take a while depending on the video length.")

    # Modified to avoid automatic extraction to MP3
    ydl_opts = {
        'format': 'bestaudio/best',
        # No postprocessors - we'll handle conversion ourselves
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'progress': True,
        'noplaylist': True  # Prevent downloading entire playlists
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            title = info.get('title', 'video')
            filename = ydl.prepare_filename(info)
            
            # Get downloaded file info
            ext = os.path.splitext(filename)[1]
            downloaded_file = os.path.join(output_dir, f"{title}{ext}")
            
            # Check if file exists with exact name
            if not os.path.exists(downloaded_file):
                # If file doesn't exist with exact name, find the downloaded file
                files = [f for f in os.listdir(output_dir) if f.endswith(ext)]
                if files:
                    downloaded_file = os.path.join(output_dir, files[0])
                else:
                    raise FileNotFoundError(f"No audio file found in {output_dir}")
            
            # Verify original downloaded file duration
            try:
                original_duration_minutes = get_audio_duration(downloaded_file)
                print(f"✅ Original downloaded file:")
                print(f"   - File: {os.path.basename(downloaded_file)}")
                print(f"   - File Size: {os.path.getsize(downloaded_file) / (1024 * 1024):.1f} MB")
                print(f"   - Duration: {original_duration_minutes:.1f} minutes")
                
                if original_duration_minutes < 60:
                    print("⚠️ WARNING: Downloaded audio is shorter than expected (< 60 min).")
                    proceed = input("Continue with this shorter file? (y/n): ")
                    if proceed.lower() not in ['y', 'yes']:
                        print("❌ Process cancelled by user.")
                        sys.exit(0)
            except Exception as e:
                print(f"⚠️ Could not verify original file duration: {e}")
            
            # Create an ASCII-safe filename for further processing
            ascii_filename = f"youtube_audio_full_{hash(downloaded_file) % 10000:04d}{ext}"
            ascii_file_path = os.path.join(output_dir, ascii_filename)
            
            # Stream-based copy to ensure no truncation
            safe_copy_file(downloaded_file, ascii_file_path)
            
            # Convert to MP3 properly
            mp3_output = os.path.splitext(ascii_file_path)[0] + ".mp3"
            success = convert_to_mp3(ascii_file_path, mp3_output)
            
            if not success:
                print("⚠️ MP3 conversion may have issues. Check the file before proceeding.")
                proceed = input("Continue anyway? (y/n): ")
                if proceed.lower() not in ['y', 'yes']:
                    print("❌ Process cancelled by user.")
                    sys.exit(0)
            
            # Get final MP3 file info
            file_size_mb = os.path.getsize(mp3_output) / (1024 * 1024)
            audio_duration_minutes = get_audio_duration(mp3_output)

            print(f"✅ Converted MP3:")
            print(f"   - File: {os.path.basename(mp3_output)}")
            print(f"   - File Size: {file_size_mb:.1f} MB")
            print(f"   - Duration: {audio_duration_minutes:.1f} minutes")

            # Final check
            if abs(audio_duration_minutes - original_duration_minutes) > 5:
                print(f"⚠️ CRITICAL WARNING: MP3 duration ({audio_duration_minutes:.1f} min) ")
                print(f"   differs significantly from original ({original_duration_minutes:.1f} min)!")
                proceed = input("Continue anyway? (y/n): ")
                if proceed.lower() not in ['y', 'yes']:
                    print("❌ Process cancelled by user.")
                    sys.exit(0)

            return mp3_output

    except Exception as e:
        logger.error(f"Error downloading or processing YouTube video: {e}")
        print("\nPossible issues:")
        print("- The video might be unavailable or restricted")
        print("- Your network connection might be unstable")
        print("- YouTube might be blocking the download")
        print("- There might be an issue with the audio processing")
        print(f"- Error details: {str(e)}")
        raise

def split_audio_into_chunks(audio_file: str, chunk_minutes: int = 10, output_dir: str = None) -> List[Dict]:
    """Split long audio file into manageable chunks using ffmpeg"""
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
        
    # Get total duration
    duration_minutes = get_audio_duration(audio_file)
    chunk_seconds = chunk_minutes * 60
    total_seconds = duration_minutes * 60
    
    # Calculate number of chunks
    num_chunks = math.ceil(total_seconds / chunk_seconds)
    
    chunk_files = []
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    
    print(f"🔄 Splitting {duration_minutes:.1f} minute audio into {num_chunks} chunks of {chunk_minutes} minutes each...")
    
    for i in range(num_chunks):
        # Calculate chunk start and duration
        start_seconds = i * chunk_seconds
        duration = min(chunk_seconds, total_seconds - start_seconds)
        
        if duration <= 0:
            break
            
        # Create output filename
        chunk_file = os.path.join(output_dir, f"{base_filename}_chunk_{i:03d}.mp3")
        
        # Run FFmpeg to extract chunk
        ffmpeg_cmd = [
            'ffmpeg', '-y', 
            '-i', audio_file, 
            '-ss', str(start_seconds), 
            '-t', str(duration),
            '-acodec', 'copy', 
            chunk_file
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Calculate precise timestamps
            chunk_start_ms = start_seconds * 1000
            chunk_end_ms = (start_seconds + duration) * 1000
            
            chunk_files.append({
                "file": chunk_file,
                "start_ms": chunk_start_ms,
                "end_ms": chunk_end_ms
            })
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating chunk {i}: {e}")
            # Continue with next chunk
    
    print(f"✅ Created {len(chunk_files)} audio chunks")
    return chunk_files

def transcribe_chunk(chunk_file: str, chunk_start_ms: int, openai_api_key: str) -> List[Dict]:
    """Transcribe a single audio chunk using OpenAI Whisper API"""
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    chunk_offset_seconds = chunk_start_ms / 1000  # Convert to seconds for timestamp adjustment
    
    # Check file size and compress if needed
    file_size = os.path.getsize(chunk_file) / (1024 * 1024)  # Convert to MB
    
    temp_audio = None
    try:
        if file_size > 25:  # OpenAI's API has a 25MB limit
            print(f"🔄 Chunk exceeds 25MB limit. Compressing...")
            
            # Create a compressed version
            temp_fd, temp_audio = tempfile.mkstemp(suffix=".mp3")
            os.close(temp_fd)
            
            # Use ffmpeg to compress
            ffmpeg_cmd = [
                'ffmpeg', '-y', 
                '-i', chunk_file,
                '-b:a', '64k',
                temp_audio
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            file_to_use = temp_audio
            
            # Get new file size
            new_file_size = os.path.getsize(file_to_use) / (1024 * 1024)
            print(f"   - Compressed from {file_size:.1f}MB to {new_file_size:.1f}MB")
        else:
            file_to_use = chunk_file
        
        # Use OpenAI's Whisper API to transcribe with timestamps
        with open(file_to_use, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="verbose_json"
            )
        
        # Extract segments with timestamps
        result = transcript.model_dump() if hasattr(transcript, 'model_dump') else transcript
        segments = result.get('segments', [])
        
        # Adjust timestamps to account for chunk position in original audio
        adjusted_segments = []
        for segment in segments:
            adjusted_segments.append({
                'start': segment.get('start', 0) + chunk_offset_seconds,
                'end': segment.get('end', 0) + chunk_offset_seconds,
                'text': segment.get('text', '').strip()
            })
        
        return adjusted_segments
        
    except Exception as e:
        print(f"❌ Error transcribing chunk: {e}")
        return []
        
    finally:
        # Clean up temporary file if created
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

def transcribe_with_timestamps(audio_file: str, openai_api_key: str, chunk_minutes: int = 10, max_minutes: int = None, skip_minutes: int = 0) -> List[Dict]:
    """Transcribe audio using OpenAI Whisper API with timestamps, chunking long files"""
    logger.info(f"Transcribing audio file: {audio_file}")
    print(f"⏱️ Transcribing audio with OpenAI Whisper... This will take a while for longer files.")

    # Get audio duration *before* any trimming (for cost estimation)
    initial_audio_duration_minutes = get_audio_duration(audio_file)

     # Create temporary directory for chunks
    temp_dir = os.path.join(os.path.dirname(audio_file), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)


    # Apply Trimming *Before* Chunking (Crucial Change)
    if (skip_minutes and skip_minutes > 0) or (max_minutes and max_minutes > 0):
        print(f"🔄 Processing audio with FFmpeg (pre-chunking)...")
        start_seconds = skip_minutes * 60
        trimmed_file = os.path.join(temp_dir, f"pre_trimmed_audio.mp3") # Use temp dir

        if max_minutes and max_minutes > 0:
            duration_seconds = max_minutes * 60
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', audio_file,
                '-ss', str(start_seconds), '-t', str(duration_seconds),
                '-acodec', 'copy', trimmed_file
            ]
            print(f"   - Skipping first {skip_minutes} minutes")
            print(f"   - Keeping {max_minutes} minutes")
        else:
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', audio_file,
                '-ss', str(start_seconds),
                '-acodec', 'copy', trimmed_file
            ]
            print(f"   - Skipping first {skip_minutes} minutes")
            print(f"   - Keeping until the end")

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_file = trimmed_file  # IMPORTANT: Use the trimmed file for subsequent steps
            audio_duration_minutes = get_audio_duration(audio_file) # Get duration of *trimmed* audio
            print(f"✅ Audio trimmed: {audio_duration_minutes:.1f} minutes kept")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error trimming audio: {e}")
            print(f"Continuing with full audio file (but chunking may still be incorrect)")
            # Here, we continue with the original audio_file, but it's a good idea
            # to log this and perhaps consider raising an exception if strict trimming is required.
            audio_duration_minutes = initial_audio_duration_minutes # Fallback to initial duration
    else:
        audio_duration_minutes = initial_audio_duration_minutes # No trimming


    # Get audio duration and estimate cost
    estimated_cost = estimate_whisper_cost(initial_audio_duration_minutes) # Use *initial* duration
    num_chunks = math.ceil(audio_duration_minutes / chunk_minutes)

    print(f"\n📊 Audio Information:")
    print(f"   - Original Duration: {initial_audio_duration_minutes:.1f} minutes")  # Show original
    print(f"   - Duration (after trimming): {audio_duration_minutes:.1f} minutes") # Show trimmed/current
    print(f"   - Will be split into {num_chunks} chunks of {chunk_minutes} minutes each")
    print(f"   - Estimated OpenAI API cost: ${estimated_cost:.2f}")

    # Ask for confirmation if the cost exceeds a certain threshold
    if estimated_cost > 5:
        proceed = input(f"\n⚠️ The estimated cost is ${estimated_cost:.2f}. Continue? (y/n): ")
        if proceed.lower() not in ['y', 'yes']:
            print("❌ Transcription cancelled by user")
            sys.exit(0)

    try:
        # Split audio into chunks
        chunk_files = split_audio_into_chunks(audio_file, chunk_minutes, temp_dir)

        # Transcribe each chunk
        all_segments = []

        for i, chunk_info in enumerate(chunk_files):
            chunk_file = chunk_info["file"]
            chunk_start_ms = chunk_info["start_ms"]

            print(f"🔄 Transcribing chunk {i+1}/{len(chunk_files)} ({os.path.basename(chunk_file)})...")

            # Transcribe the chunk
            segments = transcribe_chunk(chunk_file, chunk_start_ms, openai_api_key)
            all_segments.extend(segments)

            print(f"✅ Chunk {i+1}: Identified {len(segments)} segments")

        # Sort segments by start time to ensure they're in the correct order
        all_segments.sort(key=lambda x: x['start'])

        print(f"✅ Transcription complete! {len(all_segments)} total segments identified across {len(chunk_files)} chunks")
        return all_segments

    finally:
        # Uncomment below to clean up temporary chunks
        #If you want to keep the chunks for debugging, leave this commented out
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def segment_audio(audio_file: str, segments: List[Dict], output_dir: str) -> List[Dict]:
    """Create audio segments based on transcript segments using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"⏱️ Creating audio segments... ")
    
    dataset_items = []
    for i, segment in enumerate(segments):
        # Get segment start and end times in seconds
        start_seconds = segment['start']
        end_seconds = segment['end']
        duration_seconds = end_seconds - start_seconds
        
        # Skip segments that are too short (less than 0.1 seconds)
        if duration_seconds < 0.1:
            continue
            
        # Create segment file name
        segment_file = os.path.join(output_dir, f"segment_{i:04d}.mp3")
        
        # Use ffmpeg to extract segment
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-ss', str(start_seconds),
            '-t', str(duration_seconds),
            '-acodec', 'copy',
            segment_file
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Get letter count
            letter_count = len(segment['text'])
            
            # Add to dataset
            dataset_items.append({
                "text": segment['text'],
                "audio": segment_file,
                "start": segment['start'],
                "end": segment['end'],
                "letter_count": letter_count
            })
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating segment {i}: {e}")
            # Continue with next segment
    
    print(f"✅ Created {len(dataset_items)} audio segments")
    return dataset_items

def merge_segments_by_letter_count(segments: List[Dict], min_letters: int = 50, max_letters: int = 150) -> List[Dict]:
    """Merge segments to achieve desired letter count range"""
    print(f"⏱️ Merging segments to achieve {min_letters}-{max_letters} letter count range...")
    
    merged_segments = []
    current_text = ""
    current_audio_files = []
    current_start = 0
    current_end = 0
    current_letter_count = 0
    
    for segment in segments:
        letter_count = segment["letter_count"]
        
        # If first segment or adding won't exceed max_letters
        if current_letter_count == 0:
            current_start = segment["start"]
            current_text = segment["text"]
            current_audio_files = [segment["audio"]]
            current_end = segment["end"]
            current_letter_count = letter_count
        elif current_letter_count + letter_count <= max_letters:
            current_text += " " + segment["text"]
            current_audio_files.append(segment["audio"])
            current_end = segment["end"]
            current_letter_count += letter_count
        else:
            # Check if current segment meets minimum letter count
            if current_letter_count >= min_letters:
                merged_segments.append({
                    "text": current_text,
                    "audio_files": current_audio_files,
                    "start": current_start,
                    "end": current_end,
                    "letter_count": current_letter_count
                })
            
            # Start a new segment
            current_start = segment["start"]
            current_text = segment["text"]
            current_audio_files = [segment["audio"]]
            current_end = segment["end"]
            current_letter_count = letter_count
    
    # Add the last segment if it meets criteria
    if current_letter_count >= min_letters:
        merged_segments.append({
            "text": current_text,
            "audio_files": current_audio_files,
            "start": current_start,
            "end": current_end,
            "letter_count": current_letter_count
        })
    
    print(f"✅ Created {len(merged_segments)} merged segments in the {min_letters}-{max_letters} letter range")
    return merged_segments

def merge_audio_files(audio_files: List[str], output_file: str) -> str:
    """Merge multiple audio files using ffmpeg"""
    # Create temp file listing all input files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for audio_file in audio_files:
            f.write(f"file '{os.path.abspath(audio_file)}'\n")
        temp_file = f.name
    
    try:
        # Run ffmpeg to concatenate files
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_file,
            '-c', 'copy',
            output_file
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging audio files: {e}")
        # If concat fails, try a different approach with filter_complex
        try:
            inputs = []
            for i, audio_file in enumerate(audio_files):
                inputs.extend(['-i', audio_file])
            
            filter_complex = f'concat=n={len(audio_files)}:v=0:a=1[out]'
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                output_file
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_file
        except subprocess.CalledProcessError as e2:
            print(f"❌ Fallback merge also failed: {e2}")
            # As a last resort, use pydub but only for this small segment
            try:
                combined = AudioSegment.empty()
                for audio_file in audio_files:
                    segment = AudioSegment.from_file(audio_file)
                    combined += segment
                combined.export(output_file, format="mp3")
                return output_file
            except Exception as e3:
                print(f"❌ All merge methods failed: {e3}")
                return audio_files[0]  # Return just the first file as a fallback
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_final_dataset(merged_segments: List[Dict], output_dir: str) -> List[Dict]:
    """Create final dataset with merged audio segments"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"⏱️ Creating final dataset... ")
    
    dataset_items = []
    for i, segment in enumerate(merged_segments):
        # Merge audio files for this segment
        output_file = os.path.join(output_dir, f"merged_segment_{i:04d}.mp3")
        merge_audio_files(segment["audio_files"], output_file)
        
        dataset_items.append({
            "text": segment["text"],
            "audio": output_file,
            "letter_count": segment["letter_count"]  # Keep for statistics, but won't be included in the final dataset
        })
    
    print(f"✅ Final dataset created with {len(dataset_items)} items")
    return dataset_items

def upload_to_huggingface(dataset_items: List[Dict], dataset_name: str, hf_token: str):
    """Create and upload dataset to Hugging Face"""
    # Convert to Dataset object - only include text and audio columns
    dataset_dict = {
        "text": [item["text"] for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items]
    }
    
    dataset = Dataset.from_dict(
        dataset_dict,
        features=Features({
            "text": Value("string"),
            "audio": Audio()
        })
    )
    
    # Log in to Hugging Face
    login(token=hf_token)
    
    # Push to the Hub
    dataset.push_to_hub(dataset_name)
    logger.info(f"Dataset uploaded to Hugging Face: {dataset_name}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Create a text-audio dataset from YouTube videos")
    parser.add_argument("--video_url", type=str, help="YouTube video URL")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--min_letters", type=int, default=50, help="Minimum letters per segment")
    parser.add_argument("--max_letters", type=int, default=150, help="Maximum letters per segment")
    parser.add_argument("--ffmpeg_path", type=str, help="Path to FFmpeg bin directory (e.g., C:\\ffmpeg\\bin)")
    parser.add_argument("--max_minutes", type=int, help="Maximum length of audio in minutes (trims if longer)")
    parser.add_argument("--skip_minutes", type=int, help="Number of minutes to skip from the beginning")
    parser.add_argument("--chunk_minutes", type=int, default=10, help="Size of audio chunks for transcription in minutes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies(args.ffmpeg_path):
        print("\nℹ️ If you know FFmpeg is installed but not in your PATH, you can specify its location:")
        print("   python script.py --ffmpeg_path C:\\path\\to\\ffmpeg\\bin")
        
        # Ask user if they want to specify FFmpeg path
        user_input = input("\nDo you want to specify the FFmpeg path now? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            ffmpeg_path = input("Enter the path to your FFmpeg bin directory: ")
            if check_dependencies(ffmpeg_path):
                # Update args
                args.ffmpeg_path = ffmpeg_path
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    # If video_url not provided, ask for it
    if not args.video_url:
        args.video_url = input("Enter YouTube video URL: ")
    
    # If max_minutes not provided, ask for it
    if args.max_minutes is None:
        max_minutes_input = input("Enter maximum audio length in minutes (or press Enter for no limit): ")
        if max_minutes_input.strip():
            try:
                args.max_minutes = int(max_minutes_input)
                print(f"✅ Audio will be limited to {args.max_minutes} minutes (after skipping)")
            except ValueError:
                print("❌ Invalid input. No time limit will be applied.")
                args.max_minutes = 0
        else:
            args.max_minutes = 0
    
    # If skip_minutes not provided, ask for it
    if args.skip_minutes is None:
        skip_minutes_input = input("Enter number of minutes to skip from the beginning (or press Enter for none): ")
        if skip_minutes_input.strip():
            try:
                args.skip_minutes = int(skip_minutes_input)
                print(f"✅ Will skip the first {args.skip_minutes} minutes of audio")
            except ValueError:
                print("❌ Invalid input. Will start from the beginning.")
                args.skip_minutes = 0
        else:
            args.skip_minutes = 0
    
    # Ask for chunk size if not specified
    if not args.chunk_minutes:
        chunk_minutes_input = input("Enter chunk size in minutes for transcription (recommended: 10): ")
        if chunk_minutes_input.strip():
            try:
                args.chunk_minutes = int(chunk_minutes_input)
                print(f"✅ Audio will be processed in {args.chunk_minutes}-minute chunks")
            except ValueError:
                args.chunk_minutes = 10
                print(f"❌ Invalid input. Using default: {args.chunk_minutes}-minute chunks")
        else:
            args.chunk_minutes = 10
            print(f"✅ Using default: {args.chunk_minutes}-minute chunks")
            
    # Clarify the expected total duration if both skip and max are specified
    if args.skip_minutes and args.skip_minutes > 0 and args.max_minutes and args.max_minutes > 0:
        print(f"👉 Processing audio: Skip first {args.skip_minutes} minutes, then keep up to {args.max_minutes} minutes")
    
    # Print ASCII art banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  YouTube to Audio-Text Dataset Creator                       ║
║  ---------------------------------------                      ║
║  Create a Hugging Face dataset from YouTube videos           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Ask for OpenAI API key
    openai_api_key = input("Enter your OpenAI API key: ")
    
    # Ask for Hugging Face dataset name and token
    dataset_name = input("Enter Hugging Face dataset name (e.g., username/dataset-name): ")
    hf_token = input("Enter your Hugging Face API token: ")
    
    # Verify API keys
    if not verify_api_keys(openai_api_key, hf_token):
        logger.error("API key verification failed. Please check your keys and try again.")
        sys.exit(1)
    
    # Create output directories
    audio_dir = os.path.join(args.output_dir, "audio")
    segments_dir = os.path.join(args.output_dir, "segments")
    merged_dir = os.path.join(args.output_dir, "merged")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Main workflow
    try:
        # 1. Download YouTube audio
        logger.info(f"Downloading audio from: {args.video_url}")
        audio_file = download_youtube_audio(args.video_url, audio_dir, args.max_minutes, args.skip_minutes)
        
        # 2. Transcribe with timestamps using OpenAI API
        logger.info("Transcribing audio with OpenAI Whisper API...")
        segments = transcribe_with_timestamps(audio_file, openai_api_key, args.chunk_minutes)
        
        final_dataset_items = create_optimized_segments(
            audio_file, 
            segments, 
            merged_dir,
            min_seconds=10.0,  # 10 seconds minimum
            max_seconds=15.0   # 15 seconds maximum
        )
        
        # 6. Upload to Hugging Face
        logger.info(f"Uploading dataset to Hugging Face: {dataset_name}")
        print(f"⏱️ Uploading dataset to Hugging Face... This may take a while depending on the dataset size.")
        dataset = upload_to_huggingface(final_dataset_items, dataset_name, hf_token)
        
        # Print final summary
        print("\n" + "=" * 60)
        print(f"✅ Process completed successfully!")
        print(f"📊 Dataset Statistics:")
        print(f"   - Total segments: {len(final_dataset_items)}")
        
        # Calculate total letters
        total_letters = sum(item["letter_count"] for item in final_dataset_items)
        print(f"   - Total characters: {total_letters}")
        
        # Calculate average letters per segment
        avg_letters = total_letters / len(final_dataset_items) if final_dataset_items else 0
        print(f"   - Average characters per segment: {avg_letters:.1f}")
        
        # Calculate min and max letters in segments
        min_letters = min(item["letter_count"] for item in final_dataset_items) if final_dataset_items else 0
        max_letters = max(item["letter_count"] for item in final_dataset_items) if final_dataset_items else 0
        print(f"   - Character count range: {min_letters}-{max_letters} characters")
        
        # Calculate size of dataset
        dataset_size_mb = sum(os.path.getsize(item["audio"]) for item in final_dataset_items) / (1024 * 1024)
        print(f"   - Dataset size: {dataset_size_mb:.1f} MB")
        
        print(f"\n🔗 Your dataset is available at: https://huggingface.co/datasets/{dataset_name}")
        print(f"📝 Each row contains:")
        print(f"   - Text (50-150 characters)")
        print(f"   - Corresponding audio")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()