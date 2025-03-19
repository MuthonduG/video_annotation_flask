import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import Response
from app.transcribe import video_audio_extraction  

def tonal_analysis():
    """Extracts pitch and loudness data from a video’s audio."""
    audio_data = video_audio_extraction()  
    y, sr = librosa.load(audio_data, sr=None)

    # Extract pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    time = librosa.times_like(pitches[0], sr=sr)  # Ensure time axis matches pitch frames

    # Convert sparse pitch matrix to a single pitch array
    pitch_values = []
    pitch_times = []
    
    for i in range(pitches.shape[1]):  # Iterate over frames
        pitch_col = pitches[:, i]
        nonzero_pitch = pitch_col[pitch_col > 0]
        if len(nonzero_pitch) > 0:
            pitch_values.append(np.mean(nonzero_pitch))  # Use mean pitch per frame
            pitch_times.append(time[i])  # Store corresponding time
    
    average_pitch = np.mean(pitch_values) if pitch_values else 0

    # Extract && compute loudness (RMS Energy)
    rms_waves = librosa.feature.rms(y=y)[0]
    rms_time = librosa.times_like(rms_waves, sr=sr)

    return average_pitch, rms_waves, rms_time, pitch_values, pitch_times

def pitch_data_visualization():
    """Generates the tonal analysis plot and returns it as an in-memory image."""
    average_pitch, rms_waves, rms_time, pitch_values, pitch_times = tonal_analysis()

    # Create figure
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.scatter(pitch_times, pitch_values, color='red', s=1, label="Pitch (Hz)")
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