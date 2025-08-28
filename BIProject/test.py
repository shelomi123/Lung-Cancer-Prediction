import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Load and preprocess the data
data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')

# Standardize column names to remove spaces, convert to uppercase, and strip trailing underscores
data.columns = data.columns.str.replace(' ', '_').str.strip().str.upper().str.rstrip('_')
data['GENDER'] = data['GENDER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data.drop_duplicates(inplace=True)

# Split data into features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Feature selection with RFE
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=10, step=1)
rfe_selector.fit(X_train_resampled, y_train_resampled)
selected_features = X.columns[rfe_selector.support_].str.replace(' ', '_').str.strip().str.upper().str.rstrip('_')

# Train a Random Forest model on selected features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled[:, rfe_selector.support_], y_train_resampled)

# Streamlit UI setup
st.title("Predicting Lung Cancer")
st.write("This app uses a machine learning model to predict lung cancer based on various health factors. Please enter values for each feature below.")

# Sidebar for user input
def user_input_features():
    GENDER = st.sidebar.selectbox("Gender (1 = Male, 0 = Female)", (0, 1))
    AGE = st.sidebar.slider("Age", 0, 100, 50)
    SMOKING = st.sidebar.selectbox("Smoking (1 = Yes, 0 = No)", (0, 1))
    YELLOW_FINGERS = st.sidebar.selectbox("Yellow Fingers (1 = Yes, 0 = No)", (0, 1))
    ANXIETY = st.sidebar.selectbox("Anxiety (1 = Yes, 0 = No)", (0, 1))
    PEER_PRESSURE = st.sidebar.selectbox("Peer Pressure (1 = Yes, 0 = No)", (0, 1))
    CHRONIC_DISEASE = st.sidebar.selectbox("Chronic Disease (1 = Yes, 0 = No)", (0, 1))
    FATIGUE = st.sidebar.selectbox("Fatigue (1 = Yes, 0 = No)", (0, 1))
    ALLERGY = st.sidebar.selectbox("Allergy (1 = Yes, 0 = No)", (0, 1))
    WHEEZING = st.sidebar.selectbox("Wheezing (1 = Yes, 0 = No)", (0, 1))
    ALCOHOL_CONSUMING = st.sidebar.selectbox("Alcohol Consuming (1 = Yes, 0 = No)", (0, 1))
    COUGHING = st.sidebar.selectbox("Coughing (1 = Yes, 0 = No)", (0, 1))
    SWALLOWING_DIFFICULTY = st.sidebar.selectbox("Swallowing Difficulty (1 = Yes, 0 = No)", (0, 1))
    
    features = {
        'GENDER': GENDER,
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC_DISEASE': CHRONIC_DISEASE,
        'FATIGUE': FATIGUE,
        'ALLERGY': ALLERGY,
        'WHEEZING': WHEEZING,
        'ALCOHOL_CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SWALLOWING_DIFFICULTY': SWALLOWING_DIFFICULTY
    }
    
    return pd.DataFrame(features, index=[0])

# Get user input
input_df = user_input_features()

# Standardize input columns to match selected_features
input_df.columns = input_df.columns.str.replace(' ', '_').str.strip().str.upper().str.rstrip('_')

# Ensure input_df has all required columns by reordering to match selected_features
input_df_selected = input_df[selected_features]

# Scale input
input_df_scaled = scaler.transform(input_df_selected.reindex(columns=X.columns, fill_value=0))

# Prediction and output
if st.button("Predict"):
    prediction = rf_model.predict(input_df_scaled)
    prediction_proba = rf_model.predict_proba(input_df_scaled)
    st.write("Prediction: **{}**".format("Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"))
    st.write("Probability of Lung Cancer: {:.2f}%".format(prediction_proba[0][1] * 100))

# Model evaluation metrics
st.subheader("Model Evaluation Metrics")

# Display Confusion Matrix
#cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled[:, rfe_selector.support_]))
#fig = px.imshow(cm, text_auto=True, aspect='auto', color_continuous_scale='Blues')
#fig.update_layout(title_text="Confusion Matrix - Random Forest", xaxis_title="Predicted", yaxis_title="Actual")
#st.plotly_chart(fig)

# Feature Importance
#st.subheader("Feature Importance")
#feature_importance = pd.DataFrame({
#    'feature': selected_features,
#    'importance': rf_model.feature_importances_
#}).sort_values('importance', ascending=False)

#fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', title="Feature Importance for Random Forest")
#st.plotly_chart(fig)
