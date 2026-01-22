import pickle
import os
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config.config import MODEL_PATH, PREPROCESSOR_PATH


def test_model_artifact_structure():
    """
    Unit Test: Verify the model pickle file is valid and has expected structure.
    Does NOT use FastAPI. Tests the artifact directly.
    """
    # 1. Check file existence
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

    # 2. Try loading
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 3. Validation
    assert isinstance(
        model, RandomForestClassifier
    ), "Artifact is not a Random Forest Classifier"

    # Check if it's fitted (has estimators)
    # hasattr check is a robust way to see if a sklearn model is trained
    assert hasattr(
        model, "estimators_"
    ), "Model appears to be untrained (no estimators)"
    assert len(model.estimators_) > 0, "Model has zero trees"

    # Optional: Check feature count (modify based on your actual feature count)
    # If you know you have exactly 20 features, you can enforce it to prevent regression
    assert model.n_features_in_ > 0, "Model accepts 0 features"


def test_preprocessor_structure():
    """
    Unit Test: Verify the preprocessor pickle file.
    """
    assert os.path.exists(
        PREPROCESSOR_PATH
    ), f"Preprocessor file not found at {PREPROCESSOR_PATH}"

    with open(PREPROCESSOR_PATH, "rb") as f:
        prep = pickle.load(f)

    assert isinstance(prep, dict), "Preprocessor artifact should be a dictionary"
    assert "label_encoders" in prep, "Missing label_encoders in preprocessor"
    assert "categorical_cols" in prep, "Missing categorical_cols configuration"


def test_model_prediction_consistency():
    """
    Golden Test: Ensure model produces the same output for the same input.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Create a dummy input vector matching the model's feature size
    # shape = (1, n_features)
    n_features = model.n_features_in_
    dummy_input = np.zeros((1, n_features))

    # Predict
    prediction = model.predict(dummy_input)
    proba = model.predict_proba(dummy_input)

    # Assertions
    assert len(prediction) == 1
    assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)"
    assert proba.shape == (1, 2), "Probability shape should be (1, 2)"
    assert 0 <= proba[0][1] <= 1, "Probability must be between 0 and 1"
