import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the saved model and scaler
model_path_rf = '/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/rf_lung_cancer_model.joblib'
scaler_path = '/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/scaler.joblib'
rf_model = joblib.load(model_path_rf)
scaler = joblib.load(scaler_path)

# Define feature columns
feature_columns = [
    'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SWALLOWING_DIFFICULTY'
]

# Streamlit App
st.title("Lung Cancer Prediction App")
st.write("""
This app predicts the likelihood of a patient having lung cancer based on various health factors.
Please input the following information:
""")

# Function to get user inputs
def user_input_features():
    input_data = {}
    for feature in feature_columns:
        input_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()} (1 = Yes, 0 = No)", (0, 1))
    return pd.DataFrame(input_data, index=[0])

# Get user input
input_df = user_input_features()

# Display the input for confirmation
st.subheader("User Input:")
st.write(input_df)

# Choose Model and Threshold
model_choice = st.selectbox("Choose the model to use:", ["Random Forest", "Logistic Regression"])
threshold = st.slider("Prediction threshold for lung cancer detection (lower = more sensitive)", 0.0, 1.0, 0.3)

# Train Logistic Regression if chosen
if model_choice == "Logistic Regression":
    # Load or train Logistic Regression model if necessary
    model_path_logreg = '/mnt/data/logreg_lung_cancer_model.joblib'
    try:
        log_model = joblib.load(model_path_logreg)
    except FileNotFoundError:
        # If logistic regression model is not saved, train and save it
        X = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung_cancer_data.csv')[feature_columns]
        y = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')['LUNG_CANCER']
        X_scaled = scaler.fit_transform(X)
        log_model = LogisticRegression(class_weight='balanced', random_state=42)
        log_model.fit(X_scaled, y)
        joblib.dump(log_model, model_path_logreg)
    
    current_model = log_model
else:
    current_model = rf_model

# Prediction button
if st.button("Predict"):
    # Scale the input data
    input_df_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = current_model.predict_proba(input_df_scaled)
    lung_cancer_detected = prediction_proba[0][1] > threshold  # Apply custom threshold
    prediction = 1 if lung_cancer_detected else 0
    
    # Display the results
    st.subheader("Prediction Result:")
    result = "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer Detected"
    st.write(f"**{result}**")
    
    # Display the probability
    st.write(f"**Probability of Lung Cancer:** {prediction_proba[0][1] * 100:.2f}%")
    
    # Provide additional interpretation
    if prediction == 1:
        st.warning("The model predicts that lung cancer is likely. Please consult a medical professional for further evaluation.")
    else:
        st.success("The model predicts that lung cancer is unlikely. However, this does not replace professional medical advice.")
    
    # Display Feature Importance for Random Forest
    if model_choice == "Random Forest":
        feature_importances = rf_model.feature_importances_
        st.subheader("Feature Importances:")
        for feature, importance in zip(feature_columns, feature_importances):
            st.write(f"{feature}: {importance:.4f}")

