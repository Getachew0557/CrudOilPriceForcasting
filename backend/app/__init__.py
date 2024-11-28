import sys
import os

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for communication with React frontend

    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
