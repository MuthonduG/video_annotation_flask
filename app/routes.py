from flask import Flask
from app.transcribe import transcribe  # ✅ Correct module and function name

def register_routes(app):
    @app.route("/")
    def home():
        return "Video annotator API running"

    @app.route("/transcribe")  # ✅ Corrected route name
    def transcribe_video():
        return transcribe()  # ✅ Calls the function from transcribe.py
