import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model_path = '/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/rf_lung_cancer_model.joblib'
scaler_path = '/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/scaler.joblib'

rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define the feature columns
feature_columns = [
    'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SWALLOWING_DIFFICULTY'
]

# Streamlit UI setup
st.title("Lung Cancer Prediction App")
st.write("Enter the following health factors to predict the likelihood of lung cancer.")

# Display input fields for all features in the sidebar
def user_input_features():
    input_data = {}
    for feature in feature_columns:
        input_data[feature] = st.sidebar.selectbox(f"{feature} (1 = Yes, 0 = No)", (0, 1))
    return pd.DataFrame(input_data, index=[0])

# Get user input
input_df = user_input_features()

# Display the input for debugging purposes
st.write("Input Data for Prediction:", input_df)

# Make a prediction when the button is clicked
if st.button("Predict"):ß
    # Scale the input data
    input_df_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = rf_model.predict(input_df_scaled)
    prediction_proba = rf_model.predict_proba(input_df_scaled)
    
    # Display the results
    st.write("Prediction: **{}**".format("Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"))
    st.write("Probability of Lung Cancer: {:.2f}%".format(prediction_proba[0][1] * 100))
