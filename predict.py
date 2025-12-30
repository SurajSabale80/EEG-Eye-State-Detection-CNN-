import os
import numpy as np
import requests
from tensorflow.keras.models import load_model

# URL of the CNN model on GitHub
MODEL_URL = "https://github.com/username/repo-name/raw/main/model/cnn_model.h5"

# Download the model if not already exists locally
MODEL_PATH = "cnn_model.h5"
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

# Load the model
model = load_model(MODEL_PATH)

def predict_eye_state_batch(data):
    n_samples, n_features = data.shape
    if n_features != 14:
        raise ValueError(f"Expected 14 features per row, got {n_features}")
    data = data.reshape(n_samples, 14, 1)
    predictions = model.predict(data)
    results = ["Closed Eye" if p[0] > 0.5 else "Open Eye" for p in predictions]
    return results


