import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
BRAIN_PATH = DATA_DIR / "brain.json"

_training_enabled = False
_train_lock = Lock()


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".brain-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/brain")
def get_brain():
    if not BRAIN_PATH.exists():
        return jsonify({"present": False})
    try:
        with BRAIN_PATH.open("r") as f:
            payload = json.load(f)
        return jsonify({"present": True, **payload})
    except Exception as e:
        return jsonify({"present": False, "error": str(e)}), 500


@app.post("/api/brain")
def save_brain():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "brain" not in payload:
        return jsonify({"ok": False, "error": "invalid payload"}), 400
    try:
        _atomic_write(BRAIN_PATH, json.dumps(payload))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/brain")
def delete_brain():
    try:
        if BRAIN_PATH.exists():
            BRAIN_PATH.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _brain_stats() -> dict:
    if not BRAIN_PATH.exists():
        return {"present": False, "gen": 0, "historicalBest": 0}
    try:
        with BRAIN_PATH.open("r") as f:
            payload = json.load(f)
        return {
            "present": True,
            "gen": payload.get("gen", 0),
            "historicalBest": payload.get("historicalBest", 0),
        }
    except Exception:
        return {"present": False, "gen": 0, "historicalBest": 0}


@app.get("/api/training")
def get_training():
    with _train_lock:
        enabled = _training_enabled
    return jsonify({"enabled": enabled, **_brain_stats()})


@app.post("/api/training")
def set_training():
    global _training_enabled
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled"))
    with _train_lock:
        _training_enabled = enabled
    return jsonify({"enabled": enabled, **_brain_stats()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)
