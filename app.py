"""Flask web application for resume category classification."""

from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, render_template, request


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    """Create the Flask app, gracefully handling an untrained project."""
    flask_app = Flask(__name__)
    try:
        classifier: Any = joblib.load(model_path)
    except FileNotFoundError:
        classifier = None
        flask_app.logger.warning(
            "Model file not found at %s. Run train_model.py to create it.",
            model_path,
        )

    @flask_app.get("/")
    def index():
        error = None
        if classifier is None:
            error = "The model file is missing. Run train_model.py first to train it."
        return render_template(
            "index.html", prediction=None, error=error, resume_text=""
        )

    @flask_app.post("/")
    def predict():
        resume_text = request.form.get("resume_text", "").strip()
        if not resume_text:
            error = "Please enter resume text before predicting a category."
            prediction = None
        elif classifier is None:
            error = "The model file is missing. Run train_model.py first to train it."
            prediction = None
        else:
            prediction = str(classifier.predict([resume_text])[0])
            error = None

        return render_template(
            "index.html",
            prediction=prediction,
            error=error,
            resume_text=resume_text,
        )

    @flask_app.post("/prediction")
    def prediction():
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "JSON input required"}), 400

        resume_text = data.get("resume", "")
        if not isinstance(resume_text, str) or not resume_text.strip():
            return jsonify({"error": "Resume text is required"}), 400

        if classifier is None:
            return jsonify(
                {
                    "error": "The model file is missing. "
                    "Run train_model.py first to train it."
                }
            ), 503

        category = classifier.predict([resume_text.strip()])[0]
        return jsonify({"predicted_category": str(category)})

    @flask_app.get("/health")
    def health():
        status = "ok" if classifier is not None else "model_not_loaded"
        return jsonify(status=status)

    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
