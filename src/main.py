"""
FastAPI Application for Customer Churn Prediction.
This module sets up the API, loads model artifacts, and handles prediction requests.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from src.models import (
    CustomerData,
    BatchCustomerData,
    PredictionResponse,
    BatchPredictionResponse,
)
import pickle
import pandas as pd
import logging
from config.config import MODEL_PATH, PREPROCESSOR_PATH, LOG_FILE

# Configure Logging
# Logs are written to a file and displayed in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global storage for loaded model artifacts
artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI app.
    Loads the trained model and preprocessing steps when the application starts.
    """
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            logger.error(
                "Model artifacts not found. Please check if training has been executed."
            )
            artifacts["status"] = "failed"
        else:
            with open(MODEL_PATH, "rb") as f:
                artifacts["model"] = pickle.load(f)

            with open(PREPROCESSOR_PATH, "rb") as f:
                artifacts["preprocessing"] = pickle.load(f)

            artifacts["status"] = "loaded"
            logger.info("Model artifacts loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        artifacts["status"] = "failed"
        artifacts["error"] = str(e)

    yield

    # Cleanup resources
    artifacts.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn probability.",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Duration: {process_time:.4f}s"
    )
    return response


# Utility function to preprocess single customer data
def process_single_customer(customer: CustomerData, preprocessing_data):
    """
    Preprocess raw customer data into the format expected by the model.
    Applies label encoding and scaling.
    """
    # Convert Pydantic model to DataFrame (wrap single dict in list)
    df = pd.DataFrame([customer.model_dump()])

    # Retrieve transformation objects (loaded from artifacts)
    label_encoders = preprocessing_data["label_encoders"]
    #scaler = preprocessing_data["scaler"]
    categorical_cols = preprocessing_data["categorical_cols"]
    #numerical_cols = preprocessing_data["numerical_cols"]
    feature_columns = preprocessing_data["feature_columns"]

    # Apply Label Encoding to categorical features
    for col in categorical_cols:
        # Check if column exists in input data
        if col in df.columns:
            # Apply encoding
            le = label_encoders[col]
            
            # Safe label encoding with -1 fallback for unknown values
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                logger.warning(
                    f"Unknown value '{val}' for column '{col}' encountered. Encoding as -1."
                )
                df[col] = -1

    # Apply Scaling to numerical features
    # if numerical_cols:
        # Scale numerical features
        # df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Ensure column order matches training data
    df = df[feature_columns]

    return df


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {"status": "ok"}


@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Readiness to verify model."""
    if artifacts.get("status") == "loaded":
        return {"status": "ready", "model_status": "loaded"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "model_status": artifacts.get("status", "unknown"),
            },
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """
    for single customer churn prediction.
    """
    if artifacts.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model service is unavailable")

    try:
        # Preprocess data
        processed_df = process_single_customer(customer, artifacts["preprocessing"])

        # Execute prediction
        model = artifacts["model"]
        prediction = model.predict(processed_df)[0]
        # Get prediction probability for positive class
        probability = model.predict_proba(processed_df)[0][1]

        # Format and return response
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=int(prediction),
            churn_prediction_label="Churn" if prediction == 1 else "Active",
        )

    # Handle exceptions during prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Propagate HTTP exceptions, wrap others
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


#
@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict_churn(batch_data: BatchCustomerData):
    """
    for batch customer churn prediction.
    """
    # Check if model is loaded
    if artifacts.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model service is unavailable")

    # Initialize results list
    results = []

    # Process each customer in the batch
    try:
        # Convert request to DataFrame
        customers_dicts = [c.model_dump() for c in batch_data.customers]
        df = pd.DataFrame(customers_dicts)

        label_encoders = artifacts["preprocessing"]["label_encoders"]
        scaler = artifacts["preprocessing"]["scaler"]
        categorical_cols = artifacts["preprocessing"]["categorical_cols"]
        numerical_cols = artifacts["preprocessing"]["numerical_cols"]
        feature_columns = artifacts["preprocessing"]["feature_columns"]

        # Apply transformations
        for col in categorical_cols:
            le = label_encoders[col]
            # Safe label encoding for batch data
            # Use -1 for values not in the training set classes
            mask = df[col].isin(le.classes_)
            
            # Create a Series with default -1
            safe_encoded = pd.Series(-1, index=df.index)
            
            # Transform only known values
            if mask.any():
                safe_encoded.loc[mask] = le.transform(df.loc[mask, col])
            
            df[col] = safe_encoded

        # df[numerical_cols] = scaler.transform(df[numerical_cols])
        df = df[feature_columns]

        # Batch prediction
        model = artifacts["model"]
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        # Format results
        for pred, prob in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    churn_probability=float(prob),
                    churn_prediction=int(pred),
                    churn_prediction_label="Churn" if pred == 1 else "Active",
                )
            )

        # Return batch prediction response
        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )
