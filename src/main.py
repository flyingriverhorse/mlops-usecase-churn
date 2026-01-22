"""
FastAPI Application for Customer Churn Prediction.
This module sets up the API, loads model artifacts, and handles prediction requests.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List
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
handlers: List[logging.Handler] = [logging.StreamHandler()]

# Add file handler only for non-production environments to avoid disk I/O in containers
if os.getenv("ENVIRONMENT") != "production":
    handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# Global state storage for loaded model artifacts (Model + Preprocessor)
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

    # Shutdown: Cleanup resources
    artifacts.clear()


# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn probability.",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open to all origins for development/demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Custom middleware to log request duration and status codes for observability.
    """
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
    df = pd.DataFrame([customer.model_dump()])

    label_encoders = preprocessing_data["label_encoders"]
    categorical_cols = preprocessing_data["categorical_cols"]
    feature_columns = preprocessing_data["feature_columns"]

    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            val = df[col].iloc[0]

            # Safe label encoding: Fallback to -1 if value was not seen during training
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                logger.warning(
                    f"Unknown value '{val}' for column '{col}' encoded as -1."
                )
                df[col] = -1

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
def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a single customer.
    Runs on a thread pool to avoid blocking the main event loop.
    """
    if artifacts.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model service is unavailable")

    try:
        processed_df = process_single_customer(customer, artifacts["preprocessing"])
        model = artifacts["model"]

        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]

        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=int(prediction),
            churn_prediction_label="Churn" if prediction == 1 else "Active",
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


#
@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
def batch_predict_churn(batch_data: BatchCustomerData):
    """
    Batch prediction for multiple customers.
    Uses vectorized operations for efficient preprocessing.
    """
    if artifacts.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model service is unavailable")

    results = []

    try:
        customers_dicts = [c.model_dump() for c in batch_data.customers]
        df = pd.DataFrame(customers_dicts)

        label_encoders = artifacts["preprocessing"]["label_encoders"]
        categorical_cols = artifacts["preprocessing"]["categorical_cols"]
        feature_columns = artifacts["preprocessing"]["feature_columns"]

        for col in categorical_cols:
            le = label_encoders[col]

            # Masking: Identify known vs unknown values
            mask = df[col].isin(le.classes_)

            # Initialize column with -1 (unknown) for all rows
            safe_encoded = pd.Series(-1, index=df.index)

            # Update only the known values
            if mask.any():
                safe_encoded.loc[mask] = le.transform(df.loc[mask, col])

            df[col] = safe_encoded

        df = df[feature_columns]

        model = artifacts["model"]
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        for pred, prob in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    churn_probability=float(prob),
                    churn_prediction=int(pred),
                    churn_prediction_label="Churn" if pred == 1 else "Active",
                )
            )

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )
