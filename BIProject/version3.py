import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess the data
data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')

# Standardize column names
data.columns = data.columns.str.replace(' ', '_').str.strip().str.upper().str.rstrip('_')
data['GENDER'] = data['GENDER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data.drop_duplicates(inplace=True)

# Check class balance
st.write("Class Balance for Target Variable 'LUNG_CANCER':")
st.write(data['LUNG_CANCER'].value_counts(normalize=True))

# Use all 10 features without RFE
feature_columns = [
    'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SWALLOWING_DIFFICULTY'
]
X = data[feature_columns]
y = data['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model with adjusted parameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Calculate and display model accuracy on the test set
test_predictions = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)

# Streamlit UI setup
st.title("Predicting Lung Cancer")
st.write("This app uses a machine learning model to predict lung cancer based on selected health factors.")
st.write("Using all 10 features for prediction.")
st.write(f"Model Accuracy on Test Set: **{test_accuracy * 100:.2f}%**")

# Display input fields for all features
def user_input_features(feature_columns):
    input_data = {}
    for feature in feature_columns:
        input_data[feature] = st.sidebar.selectbox(f"{feature} (1 = Yes, 0 = No)", (0, 1))
    return pd.DataFrame(input_data, index=[0])

# Get user input dynamically based on selected features
input_df = user_input_features(feature_columns)

# Display the input for debugging purposes
st.write("Input DataFrame for Prediction:", input_df)

# Scale input data using all features each time the predict button is clicked
if st.button("Predict"):
    input_df_scaled = scaler.transform(input_df)
    
    # Display scaled input for debugging
    st.write("Scaled Input DataFrame for Prediction:", pd.DataFrame(input_df_scaled, columns=feature_columns))
    
    # Make prediction
    prediction = rf_model.predict(input_df_scaled)
    prediction_proba = rf_model.predict_proba(input_df_scaled)
    
    # Display debug info on prediction probability to trace any issues
    st.write("Prediction Probabilities (All Classes):", prediction_proba)
    
    # Display prediction results
    st.write("Prediction: **{}**".format("Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"))
    st.write("Probability of Lung Cancer: {:.2f}%".format(prediction_proba[0][1] * 100))


