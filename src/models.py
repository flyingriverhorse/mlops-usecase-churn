"""
Data models for the Customer Churn Prediction API.
This module defines Pydantic models for input validation and API responses.
"""

from pydantic import BaseModel, Field
from typing import List


# Pydantic model definitions
class CustomerData(BaseModel):
    """Pydantic model for single customer data input"""

    # class attributes(fields) with type hints representing customer features
    age: int = Field(..., description="Customer age")
    gender: str = Field(..., description="Customer gender (e.g., 'Male', 'Female')")
    tenure_months: int = Field(
        ..., ge=0, description="Number of months customer has been with the company"
    )
    monthly_charges: float = Field(
        ..., gt=0, description="Monthly amount charged to the customer"
    )
    total_charges: float = Field(
        ..., ge=0, description="Total amount charged to the customer"
    )
    contract_type: str = Field(
        ...,
        description="Contract type (e.g., 'Month-to-month', 'One year', 'Two year')",
    )
    payment_method: str = Field(
        ...,
        description=(
            "Payment method (e.g., 'Electronic check', "
            "'Bank transfer', 'Credit card')"
        ),
    )
    paperless_billing: str = Field(
        ..., description="Paperless billing (e.g., 'Yes', 'No')"
    )
    num_support_tickets: int = Field(..., ge=0, description="Number of support tickets")
    num_logins_last_month: int = Field(
        ..., ge=0, description="Number of logins last month"
    )
    feature_usage_score: float = Field(..., description="Feature usage score")
    late_payments: int = Field(..., ge=0, description="Number of late payments")
    partner: str = Field(..., description="Partner (e.g., 'Yes', 'No')")
    dependents: str = Field(..., description="Dependents (e.g., 'Yes', 'No')")
    internet_service: str = Field(
        ..., description="Internet service (e.g., 'DSL', 'Fiber optic', 'No')"
    )
    online_security: str = Field(..., description="Online security (e.g., 'Yes', 'No')")
    online_backup: str = Field(..., description="Online backup (e.g., 'Yes', 'No')")
    device_protection: str = Field(
        ..., description="Device protection (e.g., 'Yes', 'No')"
    )
    tech_support: str = Field(..., description="Tech support (e.g., 'Yes', 'No')")
    streaming_tv: str = Field(..., description="Streaming TV (e.g., 'Yes', 'No')")
    streaming_movies: str = Field(
        ..., description="Streaming movies (e.g., 'Yes', 'No')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class BatchCustomerData(BaseModel):
    """Pydantic model for batch customer data input"""

    # variable customers is a list of CustomerData instances
    customers: List[CustomerData]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customers": [
                        {
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
                        },
                        {
                            "age": 60,
                            "gender": "Female",
                            "tenure_months": 2,
                            "monthly_charges": 40.20,
                            "total_charges": 80.40,
                            "contract_type": "Month-to-month",
                            "payment_method": "Electronic check",
                            "paperless_billing": "No",
                            "num_support_tickets": 5,
                            "num_logins_last_month": 12,
                            "feature_usage_score": 3.2,
                            "late_payments": 1,
                            "partner": "No",
                            "dependents": "No",
                            "internet_service": "DSL",
                            "online_security": "No",
                            "online_backup": "No",
                            "device_protection": "No",
                            "tech_support": "No",
                            "streaming_tv": "No",
                            "streaming_movies": "No",
                        },
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""

    churn_probability: float
    churn_prediction: int
    churn_prediction_label: str


class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response"""

    predictions: List[PredictionResponse]
