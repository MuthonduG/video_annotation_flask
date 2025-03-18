import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import Response
from transcribe import video_audio_extraction  

def tonal_analysis():
    """Extracts pitch and loudness data from a video’s audio."""
    audio_data = video_audio_extraction()  
    y, sr = librosa.load(audio_data, sr=None)

    # Extract pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    time = librosa.times_like(pitches, sr=sr)  

    # Convert sparse pitch matrix to a single pitch array
    non_zero_pitches = pitches[pitches > 0]  
    average_pitch = np.mean(non_zero_pitches) if len(non_zero_pitches) > 0 else 0

    # Extract && compute loudness (RMS Energy)
    rms_waves = librosa.feature.rms(y=y)[0]
    rms_time = librosa.times_like(rms_waves, sr=sr)

    return average_pitch, rms_waves, rms_time, non_zero_pitches, time, pitches

def pitch_data_visualization():
    """Generates the tonal analysis plot and returns it as an in-memory image."""
    average_pitch, rms_waves, rms_time, non_zero_pitches, time, pitches = tonal_analysis()

    # Create figure
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, pitches.T, color='gray', alpha=0.5)
    plt.scatter(time, non_zero_pitches, color='red', s=1, label="Pitch (Hz)")
    plt.title("Pitch Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rms_time, rms_waves, color='blue', label="Loudness (RMS Energy)")
    plt.title("Loudness Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()

    # Save plot to an in-memory file (instead of saving to disk)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')  # Store figure in memory
    img_buffer.seek(0)  # Move to the beginning of the BytesIO buffer
    plt.close()  # Close the plot to free up memory

    return img_buffer  # Return in-memory file
