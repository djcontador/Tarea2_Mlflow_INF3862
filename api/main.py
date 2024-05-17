#!/usr/bin/env python
# coding: utf-8

# api/main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import pandas as pd
from src.model.predict_model import predict_model
from config import MODEL_CONFIG 
import logging 

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(filename="api_logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the data model for the prediction request
class PropertyInfo(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float
    price: float

# Define a function to validate API key
async def get_api_key(api_key: str = Header(...)):
    expected_api_key = "your_secret_api_key"  # Replace with your actual secret API key

    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Endpoint for making property value predictions with API key authentication
@app.post("/predict")
async def predict_property_value(
    property_info: PropertyInfo,
    api_key: APIKey = Depends(get_api_key)  # Use the get_api_key function to validate the API key
):
    try:
        # Convert the received data into a pandas DataFrame
        input_data = pd.DataFrame([property_info.dict()])

        # Log the input data
        logging.info(f"Received prediction request: {input_data}")

        # Make the prediction using the loaded model and provided configuration
        prediction = predict_model(input_data, **MODEL_CONFIG)

        # Log the prediction result
        logging.info(f"Prediction result: {float(prediction[0])}")

        # Return the prediction as a response
        return {"prediction": float(prediction[0])}
    except Exception as e:
        # Log any exceptions that may occur during prediction
        logging.error(f"Prediction error: {str(e)}")

        # Return an error response
        return {"error": "Prediction failed"}

