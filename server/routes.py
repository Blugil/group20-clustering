from flask import Flask, jsonify, Blueprint
from pathlib import Path
import json


api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

# please for the love of all that is good never write code like this

@api_bp.route("/analysis", methods=["GET"])
def get_analysis_json():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    JSON_DIR = DATA_DIR / "json"

    file_path = Path.joinpath(JSON_DIR, f"analysis.json")

    print(file_path)

    if not file_path.exists():
        return jsonify({"status": "test2"}), 404

    with open(file_path, "r") as file:
        json_return = json.load(file)
        return jsonify(json_return), 200

    return jsonify({"status": f"something went wrong, request could not be processed"}), 500




@api_bp.route("/vis/<string:vis>/<string:embedding>/<string:cluster>/<string:hyperparam>", methods=["GET"])
def get_vis_json(vis: str, embedding: str, cluster: str, hyperparam: str):
    # vis endpoint
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    JSON_DIR = DATA_DIR / "json"

    file_path = Path.joinpath(JSON_DIR, f"{vis}_{cluster}_{hyperparam}_{embedding}.json")

    print(file_path)

    if not file_path.exists():
        return jsonify({"status": "test2"}), 404

    with open(file_path, "r") as file:
        json_return = json.load(file)
        return jsonify(json_return), 200


    return jsonify({"status": f"something went wrong, request could not be processed"}), 500