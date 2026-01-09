# Project Summary

This document outlines the development process, technical decisions, and steps taken to build this MLOps project.

## Overview
The goal was to design an end-to-end Machine Learning pipeline to predict customer churn. The solution includes model training, a REST API for inference, and containerization for deployment.

## Step 1: Environment & Setup
I started by setting up a clean Python environment. I used `uv` for faster dependency management.

```bash
# Environment creation
uv venv

# Dependency installation
uv pip install -r requirements.txt
```

## Step 2: Model Training
executed the `train_model.py` script to train the Random Forest Classifier.

```bash
python train_model.py
```

script used `customer_churn_dataset.csv`, handled preprocessing, and trained the model.

**Model Performance:**
*   **Accuracy**: 1.0
*   **Precision**: 1.0
*   **Recall**: 1.0

*We got perfect scores which indicates that dataset is likely small or potential overfitting. with real-world data, we would expect more differences.*

After training process we had artifacts in the `models/` directory:
1.  `churn_model.pkl`: The trained classifier.
2.  `preprocessing.pkl`: The encoders and scalers needed to transform new data.
3.  `metrics.pkl`: Performance metrics for version tracking.

## Step 3: API Development (FastAPI)
I developed the inference service using FastAPI in `src/main.py`.
*   **Endpoints**: Implemented `/predict` for single requests and `/batch-predict` for bulk processing.
*   **Validation**: Used Pydantic models to validate input types.
*   **Logging**: Implemented middleware to log request details and timing to `app_activity.log` for monitoring.

## Step 4: Containerization
I containerized it using Docker.
*  **Compose**: created `docker-compose.yml` to define the service, exposing port 8000.
*  **Dockerfile**: Set up the environment with Python 3.12-slim and installed dependencies.

## How to Run
To start the service locally:
```bash
python run_api.py
# Access documentation at http://localhost:8000/docs
```