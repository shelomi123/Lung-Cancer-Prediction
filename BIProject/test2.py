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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and preprocess the data
data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')

# Standardize column names
data.columns = data.columns.str.replace(' ', '_').str.strip().str.upper().str.rstrip('_')
data['GENDER'] = data['GENDER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data.drop_duplicates(inplace=True)

# Specify all feature engineering columns
feature_columns = [
    'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
    'SWALLOWING_DIFFICULTY'
]

# Use only the specified feature engineering columns for training
X = data[feature_columns]
y = data['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature selection with RFE, limited to feature_columns
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=5, step=1)  # Adjust n_features_to_select as needed
rfe_selector.fit(X_train, y_train)
selected_features = X_train.columns[rfe_selector.support_]

# Filter X_train and X_test to only include selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train a Random Forest model on selected features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Calculate and display model accuracy on the test set
test_predictions = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)

# Streamlit UI setup
st.title("Predicting Lung Cancer")
st.write("This app uses a machine learning model to predict lung cancer based on selected health factors.")
st.write("Selected Features for Prediction:", selected_features.tolist())  # Show the selected features
st.write(f"Model Accuracy on Test Set: **{test_accuracy * 100:.2f}%**")  # Display accuracy

# Display input fields only for the selected RFE features
def user_input_features(selected_features):
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.sidebar.selectbox(f"{feature} (1 = Yes, 0 = No)", (0, 1))
    return pd.DataFrame(input_data, index=[0])

# Get user input dynamically based on selected features
input_df = user_input_features(selected_features)

# Display the input for debugging purposes
st.write("Input DataFrame for Prediction:", input_df)

# Scale input data using only the selected features each time the predict button is clicked
if st.button("Predict"):
    input_df_scaled = scaler.transform(input_df)  # Scale the input data dynamically
    
    # Display scaled input for debugging
    st.write("Scaled Input DataFrame for Prediction:", pd.DataFrame(input_df_scaled, columns=selected_features))
    
    # Make prediction
    prediction = rf_model.predict(input_df_scaled)
    prediction_proba = rf_model.predict_proba(input_df_scaled)
    
    # Display prediction results
    st.write("Prediction: **{}**".format("Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"))
    st.write("Probability of Lung Cancer: {:.2f}%".format(prediction_proba[0][1] * 100))

# Model evaluation metrics
st.subheader("Model Evaluation Metrics")

# Display Confusion Matrix
cm = confusion_matrix(y_test, test_predictions)
fig = px.imshow(cm, text_auto=True, aspect='auto', color_continuous_scale='Blues')
fig.update_layout(title_text="Confusion Matrix - Random Forest", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(fig)

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', title="Feature Importance for Random Forest")
st.plotly_chart(fig)
