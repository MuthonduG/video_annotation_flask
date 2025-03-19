from app.transcribe import video_audio_extraction, transcribe
from app.pitch_analysis import tonal_analysis
import numpy as np
import json

def segment_audio_with_tone():
    """Segments audio by integrating transcription with tonal variations."""
    audio_data = video_audio_extraction()
    if isinstance(audio_data, dict) and "error" in audio_data:
        return {"error": "Failed to extract audio"}  # Return a JSON-compatible error

    avg_pitch, rms_energy, *_ = tonal_analysis()
    
    transcript_segments = transcribe()
    if isinstance(transcript_segments, str):
        try:
            transcript_segments = json.loads(transcript_segments)  # Convert string to Python list
        except json.JSONDecodeError:
            return {"error": "Invalid transcription format"}

    if not isinstance(transcript_segments, list):  # Ensure it's a list of dicts
        return {"error": "Unexpected transcription format"}

    # Define tone thresholds
    PITCH_THRESHOLD = 200  
    LOUDNESS_THRESHOLD = 0.05  

    labeled_segments = []
    
    for seg in transcript_segments:
        if not isinstance(seg, dict) or not all(k in seg for k in ["start", "end", "text"]):
            continue  # Skip invalid segments

        try:
            start, end, text = int(seg["start"]), int(seg["end"]), seg["text"]
        except (ValueError, TypeError):
            continue  # Skip if parsing fails

        # Ensure index range does not exceed the array length
        if start >= len(avg_pitch) or end >= len(avg_pitch):
            continue  

        avg_seg_pitch = np.mean(avg_pitch[start:end]) if len(avg_pitch[start:end]) > 0 else 0
        avg_seg_loudness = np.mean(rms_energy[start:end]) if len(rms_energy[start:end]) > 0 else 0

        label = "Emphasized Speech" if avg_seg_pitch > PITCH_THRESHOLD or avg_seg_loudness > LOUDNESS_THRESHOLD else "Normal Speech"

        labeled_segments.append({
            "start": start,
            "end": end,
            "text": text,
            "label": label
        })

    return labeled_segments
