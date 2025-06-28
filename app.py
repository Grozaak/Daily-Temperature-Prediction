import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Temperature Forecast", layout="centered")
st.title(" Next Day Temperature Predictor")

# --- Load model and scaler ---
@st.cache_resource
def load_model_file():
    model = load_model("model.h5", compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model_file()
scaler = load_scaler()

# --- File Upload Section ---
uploaded_file = st.file_uploader(" Upload your CSV file (with 'Date' and 'Temp' columns):", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df.dropna(subset=['Temp'], inplace=True)

        # --- Display dataset overview ---
        st.subheader(" Uploaded Data Preview")
        st.line_chart(df['Temp'])

        last_date = df.index[-1]
        next_day = last_date + pd.Timedelta(days=1)

        st.subheader(" Prediction Info")
        st.info(f"Last date in uploaded data: **{last_date.date()}**\nPrediction for: **{next_day.date()}**")

        # --- Preprocess uploaded data ---
        data_scaled = scaler.transform(df['Temp'].values.reshape(-1, 1))
        seq_length = 30

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len])
                y.append(data[i + seq_len])
            return np.array(X), np.array(y)

        X, y = create_sequences(data_scaled, seq_length)
        if len(X) == 0:
            st.error(" Not enough data to create sequences. At least 30 rows are required.")
            st.stop()

        # --- Prediction ---
        split_index = int(len(X) * 0.8)
        X_test = X[split_index:]
        y_test = y[split_index:]

        y_pred_scaled = model.predict(X_test)
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test)

        # --- Plot Actual vs Predicted ---
        st.subheader(" Actual vs Predicted Temperatures")
        fig, ax = plt.subplots()
        ax.plot(y_test_actual, label='Actual Temp', color='blue')
        ax.plot(y_pred, label='Predicted Temp', color='orange')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        st.pyplot(fig)

        # --- Predict Next Day ---
        last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
        next_temp_scaled = model.predict(last_sequence)
        next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
        next_temp = scaler.inverse_transform(next_temp_scaled)

        st.subheader(" Predicted Temperature for Next Day")
        st.success(f" {next_temp[0][0]:.2f} °C")

    except Exception as e:
        st.error(f" Error processing the uploaded file: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")
