#!/usr/bin/env python
# coding: utf-8

# src/model/train_model.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from src.database import DatabaseConnector
import os
import warnings
import joblib

def load_data(train_path: str, test_path: str, train_query: str = None, test_query: str = None, database_connector: DatabaseConnector = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the train and test data from the database or CSV files.

    Args:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the test data CSV file.
        train_query (str, optional): SQL query to fetch training data.
        test_query (str, optional): SQL query to fetch test data.
        database_connector (DatabaseConnector, optional): Database connection object.

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): Loaded train and test data.
    """
    # Check for invalid file paths
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError("Invalid file paths provided.")

    if database_connector:  ################### TBD
        # Database-related checks
        if not all(key in database_connector.connection_params for key in ['user', 'password', 'host', 'database']):
            raise ValueError("Invalid or incomplete database connection parameters.")

        # Fetch data from the client database
        train = database_connector.fetch_data(train_query) if train_query else pd.DataFrame()
        test = database_connector.fetch_data(test_query) if test_query else pd.DataFrame()
        
        # Check for NaN values in loaded dataframes
        if train.isna().any().any() or test.isna().any().any():
            warnings.warn("NaN values detected in the loaded data. Please handle missing values appropriately.")
        
    else:
        # Load data from CSV files
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
            
        # Check for missing paths
        if not train_path or not test_path:
            raise ValueError("Provide both train_path and test_path, or a DatabaseConnector.")
            
        # Check for NaN values in loaded dataframes
        if train.isna().any().any() or test.isna().any().any():
            warnings.warn("NaN values detected in the loaded data. Please handle missing values appropriately.")
            
    print("(src/model/train_model.py): data loaded")
    return train, test


def train_model(train: pd.DataFrame, target: str) -> Pipeline:
    """
    Trains a machine learning model.

    Args:
        train (pd.DataFrame): Training data.
        target (str): Target variable.

    Returns:
        Pipeline: Trained machine learning model.
    """
    train_cols = [col for col in train.columns if col not in ['id', target]]
    categorical_cols = ["type", "sector"]
    
    # Check if categorical_cols are present in train dataset
    missing_categorical_cols = set(categorical_cols) - set(train.columns)
    if missing_categorical_cols:
        raise ValueError(f"Missing categorical columns in the training data: {', '.join(missing_categorical_cols)}")

    categorical_transformer = TargetEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_cols)
        ]
    )

    steps = [
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(**{
            "learning_rate": 0.01,
            "n_estimators": 300,  
            "max_depth": 5,
            "loss": "absolute_error"
        }))
    ]

    pipeline = Pipeline(steps)
    
    try:
        pipeline.fit(train[train_cols], train[target])
        
        try:
            # Save the trained pipeline for deployment
            joblib.dump(pipeline, 'property_valuation_pipeline.joblib')
            print("(src/model/train_model.py): Saved trained pipeline to property_valuation_pipeline.joblib")
        except Exception as e:
            print(f"Error saving trained pipeline: {str(e)}")
        
    except Exception as e:
        raise RuntimeError(f"Error during model training: {str(e)}")
        
    return pipeline

