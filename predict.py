import os
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cnn_model.h5")  # make sure model folder exists
model = load_model(MODEL_PATH)

def predict_eye_state_batch(data):
    """
    Input: numpy array of shape (n_samples, 14)
    Output: list of predictions ("Open Eye"/"Closed Eye")
    """
    n_samples, n_features = data.shape
    if n_features != 14:
        raise ValueError(f"Expected 14 features per row, got {n_features}")

    data = data.reshape(n_samples, 14, 1)
    predictions = model.predict(data)

    results = ["Closed Eye" if p[0] > 0.5 else "Open Eye" for p in predictions]
    return results

