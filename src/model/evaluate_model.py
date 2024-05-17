#!/usr/bin/env python
# coding: utf-8

# src/model/evaluate_model.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def evaluate_model(model, test_predictions, test_target):
    """
    Evaluates the trained machine learning model.

    Args:
        model: Trained machine learning model.
        test_predictions (numpy.ndarray): Predictions on test data.
        test_target (numpy.ndarray): True target values from the test data.
    """
    print_metrics(test_predictions, test_target)
    print("(src/model/evaluate_model.py): model evaluated")


def print_metrics(predictions, target):
    """
    Prints evaluation metrics for the predictions.

    Args:
        predictions (numpy.ndarray): Predicted values.
        target (numpy.ndarray): True target values.
    """
    print("RMSE: ", np.sqrt(mean_squared_error(predictions, target)))
    print("MAPE: ", mean_absolute_percentage_error(predictions, target))
    print("MAE : ", mean_absolute_error(predictions, target))

