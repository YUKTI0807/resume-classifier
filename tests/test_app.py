from pathlib import Path

import pytest

from app import create_app


class ExampleClassifier:
    def predict(self, resumes: list[str]) -> list[str]:
        assert len(resumes) == 1
        assert isinstance(resumes[0], str)
        return ["IT"]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("app.joblib.load", lambda _path: ExampleClassifier())
    flask_app = create_app(tmp_path / "model.pkl")
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_home_page_returns_200(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Resume Category Classifier" in response.data


def test_health_endpoint_returns_json(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {"status": "ok"}


def test_prediction_route_displays_category(client):
    response = client.post(
        "/", data={"resume_text": "Experienced Python software developer"}
    )

    assert response.status_code == 200
    assert b"Predicted category" in response.data
    assert b">IT<" in response.data


def test_json_prediction_route_returns_category(client):
    response = client.post(
        "/prediction",
        json={
            "resume": "  Python developer with machine learning and data science skills  "
        },
    )

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {"predicted_category": "IT"}


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"resume": ""},
        {"resume": "   "},
        {"resume": 123},
    ],
)
def test_json_prediction_route_rejects_invalid_input(client, data):
    if data is None:
        response = client.post(
            "/prediction",
            data="not json",
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.get_json() == {"error": "JSON input required"}
    else:
        response = client.post("/prediction", json=data)
        assert response.status_code == 400
        assert response.get_json() == {"error": "Resume text is required"}


def test_empty_resume_text_displays_validation_error(client):
    response = client.post("/", data={"resume_text": "   "})

    assert response.status_code == 200
    assert b"Please enter resume text" in response.data


def test_missing_model_is_explained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def missing_model(_path: Path):
        raise FileNotFoundError

    monkeypatch.setattr("app.joblib.load", missing_model)
    flask_app = create_app(tmp_path / "missing-model.pkl")
    flask_app.config.update(TESTING=True)
    client = flask_app.test_client()

    response = client.get("/")
    post_response = client.post("/", data={"resume_text": "A software developer"})
    api_response = client.post(
        "/prediction", json={"resume": "A software developer"}
    )

    assert response.status_code == 200
    assert b"Run train_model.py first" in response.data
    assert post_response.status_code == 200
    assert b"Run train_model.py first" in post_response.data
    assert api_response.status_code == 503
    assert api_response.get_json() == {
        "error": "The model file is missing. Run train_model.py first to train it."
    }
