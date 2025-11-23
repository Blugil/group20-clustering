from flask import Flask, jsonify, Blueprint
from pathlib import Path
from routes import api_bp
import json


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        APP_NAME="vis_api",
        JSON_AS_ASCII=False,
    )


    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    @app.route("/", methods=["GET"])
    def root():
        return {'status': 'ok'}, 200
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=3000, debug=True)