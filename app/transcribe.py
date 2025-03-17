import os
import ffmpeg
from faster_whisper import WhisperModel

# Define paths
VIDEO_PATH = "assets/ray_williams.mp4"
AUDIO_DIR = "audio-assets"

# Ensure the audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

def transcribe():
    if os.path.exists(VIDEO_PATH):
        return run()
    else:
        return {"error": "Video path does not exist"}

def video_audio_extraction():
    """Extracts audio from the video file and saves it to AUDIO_DIR."""
    audio_filename = os.path.join(AUDIO_DIR, "ray_williams.wav")
    
    try:
        video_stream = ffmpeg.input(VIDEO_PATH)
        video_stream = ffmpeg.output(video_stream, audio_filename, format='wav', acodec='pcm_s16le')
        ffmpeg.run(video_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        if not os.path.exists(audio_filename):
            raise FileNotFoundError(f"Failed to extract audio: {audio_filename} not found.")
        
        return audio_filename
    except Exception as e:
        return {"error": f"Audio extraction failed: {str(e)}"}

def audio_transcription(audio_path):
    """Transcribes extracted audio using Whisper."""
    model = WhisperModel("small",  compute_type="float32")
    
    try:
        segments, info = model.transcribe(audio_path)
        language = info.language
        
        transcriptions = [
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            for segment in segments
        ]
        
        return {"language": language, "transcriptions": transcriptions}
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

def run():
    """Runs the complete process: extract audio and transcribe it."""
    audio_path = video_audio_extraction()
    
    if isinstance(audio_path, dict) and "error" in audio_path:
        return audio_path  # Return error message if extraction fails
    
    return audio_transcription(audio_path)
