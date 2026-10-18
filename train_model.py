"""Train and save the demonstration resume category classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "resumes.csv"
MODEL_PATH = PROJECT_DIR / "model.pkl"
REQUIRED_COLUMNS = {"Resume", "Category"}


def train_model(
    dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH
) -> None:
    """Train the text classifier and save its complete pipeline."""
    try:
        data = pd.read_csv(dataset_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read dataset '{dataset_path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}.")
    if data.empty:
        raise ValueError("Dataset contains no rows.")

    for column in ("Resume", "Category"):
        values = data[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset column '{column}' contains missing or empty values.")
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Dataset column '{column}' must contain text values.")

    # This tiny dataset demonstrates the workflow; it is not enough for reliable
    # real-world accuracy.
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )
    pipeline.fit(data["Resume"], data["Category"])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model trained and saved to {model_path}")


def main() -> None:
    try:
        train_model()
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
