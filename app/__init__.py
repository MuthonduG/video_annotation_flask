from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register blueprints/routes
    from app import routes
    routes.register_routes(app)

    return app
