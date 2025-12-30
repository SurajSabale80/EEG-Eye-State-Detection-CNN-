import streamlit as st
import pandas as pd
from predict import predict_eye_state_batch

st.title("👁️ EEG Eye State Detection (CNN)")

st.write("Upload a CSV file with EEG data. The app will predict the eye state for the first 10 samples.")

uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Only numeric columns
    df_numeric = df.select_dtypes(include='number')

    # Take first 10 rows and first 14 columns
    df_to_predict = df_numeric.iloc[:10, :14]

    st.write("Preview of first 10 EEG samples (14 features per row):")
    st.dataframe(df_to_predict)

    if st.button("Predict Eye State"):
        predictions = predict_eye_state_batch(df_to_predict.values)
        df_to_predict["Prediction"] = predictions
        st.success("Prediction completed for first 10 samples")
        st.dataframe(df_to_predict)

        # Optional: Download results
        st.download_button(
            label="Download Predictions",
            data=df_to_predict.to_csv(index=False),
            file_name="predictions.csv"
        )

