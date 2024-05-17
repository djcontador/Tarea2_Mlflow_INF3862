#!/usr/bin/env python
# coding: utf-8

# api/api.py

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
import logging
import joblib
import pandas as pd
from pathlib import Path
import uvicorn

app = FastAPI()

# Define a logger for API logs
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# Create a log file handler
handler = logging.FileHandler("api_logs.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define a dictionary to store valid API keys (replace with actual keys)
valid_api_keys = {"your_api_key": "user1", "another_api_key": "user2"}

# API Key Authentication Middleware
def api_key_auth(api_key: str = Header(None)):
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# API Request Models   # train_cols
class PropertyInfo(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float
    #price: float   

class PropertyValuation(BaseModel):
    # target_variable
    estimated_price: float  
        
# Define the full path for the model file
root_path = Path(__file__).parent.parent
model_file_path = root_path / 'property_valuation_pipeline.joblib'

    
# Load the model
try:
    model = joblib.load(model_file_path)
    print("(api/api.py): Loaded pre-trained model successfully.")
except FileNotFoundError:
    raise RuntimeError("Model file not found")    
     
# Endpoint for Property Valuation Predictions
@app.post("/predict", response_model=PropertyValuation)
def predict_property_valuation(property_info: PropertyInfo, api_key: str = Depends(api_key_auth)):
    try:
        # Prepare API data for prediction
        input_data = pd.DataFrame([property_info.dict()])
        prediction = model.predict(input_data)[0]

        # Log the prediction request
        logger.info(f"Property Valuation Prediction Request - Property Info: {property_info}, User: {valid_api_keys.get(api_key)}")

        return {"estimated_price": prediction}
    except Exception as e:
        logger.error(f"Error during property valuation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)