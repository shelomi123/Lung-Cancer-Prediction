import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('/Users/shelomi/Documents/UNITEC/Data Analytics and Intelligence/Assignment2/lung cancer data.csv')
data.head()

st.title('Predicting Lung Cancer')
#data.info()
#print(data.head())


#print(data.describe())

# Strip whitespace from column names
print('Strip whitespace from column names')
data.columns = data.columns.str.strip()

print('Clean "GENDER" and "LUNG_CANCER" columns')
data['GENDER'] = data['GENDER'].str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].str.strip()

print('Replace numeric values')
data.replace({2: 1, 1: 0}, inplace=True)

print('Map categorical values to binary for modeling')
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})

#print(data.head())

print('Check for missing values')
print(data.isna().sum())

print('Check for duplicates and remove them')
print(data.duplicated().sum())
print(data.drop_duplicates(inplace=True))

print('Check the class balance in the target variable')
print("\nClass balance:")
print(data['LUNG_CANCER'].value_counts(normalize=True))


#Train-Test Split and Feature Scaling
# Splitting data into features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Handling Imbalance with SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Feature selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=10, step=1)
rfe_selector = rfe_selector.fit(X_train_resampled, y_train_resampled)
selected_features = X.columns[rfe_selector.support_]

# Display selected features
print("\nSelected features:", selected_features)

# Transforming data to keep only selected features
X_train_selected = X_train_resampled[:, rfe_selector.support_]
X_test_selected = X_test_scaled[:, rfe_selector.support_]

# Define models to train
models = {
    #'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    #'SVM': SVC(random_state=42),
    #'XGBoost': XGBClassifier(random_state=42)
}

# Train models and display results
for name, model in models.items():
    model.fit(X_train_selected, y_train_resampled)
    y_pred = model.predict(X_test_selected)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    

    
    # Visualize the confusion matrix for Random Forest model
#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)
#fig = px.imshow(cm, text_auto=True, aspect='auto', color_continuous_scale='Blues')
#fig.update_layout(title_text=f'Confusion Matrix - {name}', xaxis_title='Predicted', yaxis_title='Actual')
#fig.show()

#Feature Importance (Random Forest)
# Plot feature importance for the Random Forest model
#rf_model = models['Random Forest']
#feature_importance = pd.DataFrame({
#    'feature': selected_features,
#    'importance': rf_model.feature_importances_
#}).sort_values('importance', ascending=False)

# Bar plot for feature importnace
#fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', title='Feature Importance for Random Forest')
#fig.show()

