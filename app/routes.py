from flask import Flask, send_file, jsonify
from app.transcribe import transcribe  
from app.pitch_analysis import pitch_data_visualization
from app.annonator import segment_audio_with_tone

def register_routes(app):
    @app.route("/")
    def home():
        return "Video annotator API running"

    @app.route("/transcribe") 
    def transcribe_video():
        return transcribe()  
    
    @app.route("/tone") 
    def audio_tonal_analysis():
        img_buffer = pitch_data_visualization()  # Generate plot in memory
        return send_file(img_buffer, mimetype='image/png')  # Send as response
    
    @app.route("/anonnate")
    def video_annotator():
        annotated_segments = segment_audio_with_tone()
        return jsonify(annotated_segments)  # Return JSON response
