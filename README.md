# Eye State Detection using EEG and CNN
-Live App Link:-https://7nnhgfra9yg9dfybwbg8g2.streamlit.app/
This project predicts whether a person's eye is **open** or **closed** using EEG signals and a Convolutional Neural Network (CNN).

## Features
- EEG-based eye state classification
- CSV file upload support
- Predicts first 10 samples from CSV
- Streamlit web interface
- Download predictions as CSV

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- Streamlit
- NumPy

## How to Run

1. Clone the repository
2. Place `cnn_model.h5` inside the `model/` folder
3. Create a virtual environment and activate it
4. Install requirements:

```bash
pip install -r requirements.txt
