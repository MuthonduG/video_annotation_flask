from flask import Flask

def register_routes(app):
    @app.route("/")
    def home():
        return("Video annonator API running")