"""
Data models for the Customer Churn Prediction API.
This module defines Pydantic models for input validation and API responses.
"""

from pydantic import BaseModel, Field
from typing import Literal, List


class CustomerData(BaseModel):
    """Pydantic model for single customer data input"""

    age: int = Field(..., description="Customer age")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    tenure_months: int = Field(
        ..., ge=0, description="Number of months customer has been with the company"
    )
    monthly_charges: float = Field(
        ..., gt=0, description="Monthly amount charged to the customer"
    )
    total_charges: float = Field(
        ..., ge=0, description="Total amount charged to the customer"
    )
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal["Electronic check", "Bank transfer", "Credit card"]
    paperless_billing: Literal["Yes", "No"]
    num_support_tickets: int = Field(..., ge=0)
    num_logins_last_month: int = Field(..., ge=0)
    feature_usage_score: float = Field(...)
    late_payments: int = Field(..., ge=0)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    internet_service: Literal["DSL", "Fiber optic"]
    online_security: Literal["Yes", "No"]
    online_backup: Literal["Yes", "No"]
    device_protection: Literal["Yes", "No"]
    tech_support: Literal["Yes", "No"]
    streaming_tv: Literal["Yes", "No"]
    streaming_movies: Literal["Yes", "No"]

    class Config:
        """Configuration for the CustomerData model."""

        json_schema = {
            "example": {
                "age": 45,
                "gender": "Male",
                "tenure_months": 24,
                "monthly_charges": 79.85,
                "total_charges": 1916.40,
                "contract_type": "Two year",
                "payment_method": "Credit card",
                "paperless_billing": "Yes",
                "num_support_tickets": 2,
                "num_logins_last_month": 42,
                "feature_usage_score": 8.5,
                "late_payments": 0,
                "partner": "Yes",
                "dependents": "No",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "Yes",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
            }
        }


class BatchCustomerData(BaseModel):
    """Pydantic model for batch customer data input"""

    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""

    churn_probability: float
    churn_prediction: int
    churn_prediction_label: str


class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response"""

    predictions: List[PredictionResponse]
