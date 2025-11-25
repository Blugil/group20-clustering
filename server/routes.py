from flask import Flask, jsonify, Blueprint, g
from pathlib import Path
import json
import sqlite3


def get_db(name: str = "newsgroups.db"):

    DB_DIR = Path(__file__).resolve().parents[1] / "db"
    DB_PATH = DB_DIR / name

    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@api_bp.route("/docs/<int:doc_id>")
def get_doc(doc_id: int):
    db = get_db()

    cursor = db.execute(
        "SELECT id, subject, body FROM documents WHERE id = ?",
        (doc_id,),)
    row = cursor.fetchone()
    if row is None:
        return jsonify({"status": 404}), 404
    return jsonify(dict(row)), 200



# please for the love of all that is good never write code like this

@api_bp.route("/analysis", methods=["GET"])
def get_analysis_json():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    JSON_DIR = DATA_DIR / "json"

    file_path = Path.joinpath(JSON_DIR, f"analysis.json")

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

    if not file_path.exists():
        return jsonify({"status": "test2"}), 404

    with open(file_path, "r") as file:
        json_return = json.load(file)
        return jsonify(json_return), 200


    return jsonify({"status": f"something went wrong, request could not be processed"}), 500