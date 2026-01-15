import pytest
import pandas as pd
import numpy as np
import os
from train_model import preprocess_data, train_model

# we verify function with fixture to prevent copy-paste of dummy data
@pytest.fixture
def dummy_data():
    """Create a small dummy dataset for testing"""
    data = {
        'customer_id': ['1', '2', '3', '4', '5'],
        'age': [25, 30, 35, 40, 45],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'tenure_months': [12, 24, 6, 48, 60],
        'monthly_charges': [50.0, 70.0, 40.0, 90.0, 100.0],
        'total_charges': [600.0, 1680.0, 240.0, 4320.0, 6000.0],
        'contract_type': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Two year'],
        'payment_method': ['Electronic check', 'Mailed check', 'Electronic check', 'Credit card', 'Bank transfer'],
        'paperless_billing': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'num_support_tickets': [1, 0, 3, 0, 1],
        'num_logins_last_month': [10, 20, 5, 25, 30],
        'feature_usage_score': [3.5, 4.0, 2.0, 4.5, 5.0],
        'late_payments': [0, 1, 2, 0, 0],
        'partner': ['No', 'Yes', 'No', 'Yes', 'Yes'],
        'dependents': ['No', 'Yes', 'No', 'Yes', 'Yes'],
        'internet_service': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'No'],
        'online_security': ['No', 'Yes', 'No', 'Yes', 'No important'],
        'online_backup': ['Yes', 'No', 'No', 'Yes', 'No important'],
        'device_protection': ['No', 'Yes', 'No', 'Yes', 'No important'],
        'tech_support': ['No', 'Yes', 'No', 'Yes', 'No important'],
        'streaming_tv': ['Yes', 'Yes', 'No', 'Yes', 'No important'],
        'streaming_movies': ['No', 'Yes', 'No', 'Yes', 'No important'],
        'churn': [0, 0, 1, 0, 0]
    }
    return pd.DataFrame(data)

def test_preprocess_data(dummy_data):
    """Test that preprocessing correctly handles data and splits X, y"""
    X, y, preprocessing = preprocess_data(dummy_data)
    
    # Check that customer_id is dropped
    assert 'customer_id' not in X.columns
    
    # Check that churn is separated as target
    assert 'churn' not in X.columns
    assert len(y) == 5
    
    # Check that preprocessing dictionary contains expected keys
    assert "label_encoders" in preprocessing
    assert "categorical_cols" in preprocessing
    assert "feature_columns" in preprocessing

def test_train_model_execution(dummy_data):
    """Test that training runs without error and produces a model"""
    X, y, preprocessing = preprocess_data(dummy_data)
    
    # Train the model with the dummy processed data
    # here we pass the whole dummy X as training data for valid syntax check)
    model = train_model(X, y, preprocessing["label_encoders"])
    
    # Check if model object is created and has predictable attribute
    assert hasattr(model, 'predict')
    # Check if it is a fitted estimator
    assert hasattr(model, 'estimators_')
    
    # Try a prediction
    preds = model.predict(X)
    assert len(preds) == len(X)
