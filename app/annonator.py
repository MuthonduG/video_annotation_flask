from transcribe import video_audio_extraction, transcribe
from pitch_analysis import tonal_analysis


def segment_audio_with_tone():
    """Segments audio by integrating transcription with tonal variations."""
    audio_data = video_audio_extraction()
    if isinstance(audio_data, dict) and "error" in audio_data:
        return audio_data  # Return error if extraction fails

    avg_pitch, rms_energy = tonal_analysis(audio_data)
    transcript_segments = transcribe(audio_data)

    # Define tone thresholds
    PITCH_THRESHOLD = 200  
    LOUDNESS_THRESHOLD = 0.05  

    labeled_segments = []
    
    for seg in transcript_segments:
        start, end, text = seg["start"], seg["end"], seg["text"]

        # Get average pitch & loudness for the time range
        avg_seg_pitch = np.mean(avg_pitch[int(start):int(end)])
        avg_seg_loudness = np.mean(rms_energy[int(start):int(end)])

        # Label based on tonal emphasis
        label = "Emphasized Speech" if avg_seg_pitch > PITCH_THRESHOLD or avg_seg_loudness > LOUDNESS_THRESHOLD else "Normal Speech"

        labeled_segments.append({
            "start": start,
            "end": end,
            "text": text,
            "label": label
        })

    return labeled_segments
