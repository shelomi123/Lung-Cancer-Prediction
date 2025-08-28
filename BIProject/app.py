from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

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

# Initialize the FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data
class InputData(BaseModel):
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SWALLOWING_DIFFICULTY: int

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()], columns=feature_columns)
    
    # Scale the input data
    input_df_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = rf_model.predict(input_df_scaled)
    prediction_proba = rf_model.predict_proba(input_df_scaled)
    
    # Format the response
    response = {
        "Prediction": "Lung Cancer" if prediction[0] == 1 else "No Lung Cancer",
        "Probability of Lung Cancer": prediction_proba[0][1] * 100
    }
    return response
