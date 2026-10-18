import os
import joblib
import numpy as np
import pandas as pd

def test_dataset_exists():
    """Verify that the cleaned dataset exists and is non-empty."""
    assert os.path.exists("data/raw/heart.csv"), "Cleaned dataset not found!"
    df = pd.read_csv("data/raw/heart.csv")
    assert not df.empty, "Dataset is empty!"
    assert 'target' in df.columns, "Target column missing from dataset!"

def test_model_artifacts_exist():
    """Verify that model and scaler artifacts are successfully saved."""
    assert os.path.exists("model.pkl"), "Trained model artifact (model.pkl) missing!"
    assert os.path.exists("scaler.pkl"), "Scaler artifact (scaler.pkl) missing!"

def test_model_inference():
    """Verify that the saved model and scaler can perform a prediction successfully."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    dummy_input = np.array([[63.0, 1.0, 3.0, 145.0, 233.0, 1.0, 0.0, 150.0, 0.0, 2.3, 0.0, 0.0, 1.0]])
    
    scaled_input = scaler.transform(dummy_input)
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)
    
    assert len(prediction) == 1, "Prediction output length mismatch!"
    assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)!"
    assert probabilities.shape == (1, 2), "Probability shape should be (1, 2)!"
