#!/usr/bin/env python
# coding: utf-8

# src/model/predict_model.py

def predict_model(model, input_data):
    """
    Makes predictions using the trained machine learning model.

    Args:
        model: Trained machine learning model.
        input_data (pd.DataFrame): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    print("(src/model/predict_model.py): the model made the prediction")
    return model.predict(input_data)

