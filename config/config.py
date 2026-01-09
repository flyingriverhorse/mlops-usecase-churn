import os

# This is where our project lives
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where we keep the trained model files
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessing.pkl")

# Logging settings
LOG_FILE = "app_activity.log"
