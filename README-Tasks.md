# MLOps Use Case: Customer Churn Prediction

## Use Case
Predicting customer churn for a subscription-based service

## Objective
Design and implement an end-to-end MLOps pipeline to deploy and manage a machine learning model for customer churn prediction. The solution should:
- **Train a model** using the provided dataset and training script
- Build a **FastAPI** REST API for predictions
- Use **Docker** for containerization
- Deploy using **either Kubernetes OR Docker Compose** (choose one)
- Follow **MLOps best practices** to ensure scalability, reliability, and reproducibility
- **Bonus:** Document your approach for implementing a CI/CD pipeline

---

## Task Breakdown

### 1. Model Training

**Scenario:**  
Train a machine learning model using the provided customer churn dataset (`customer_churn_dataset.csv`) and training script (`train_model.py`). The script trains a Random Forest Classifier and saves the model artifacts for deployment.

**Task:**
- Set up Python environment and install dependencies from `requirements.txt`
- Review the training script (`train_model.py`) to understand the preprocessing and training pipeline
- Run the training script to generate model artifacts:
  ```bash
  python train_model.py
  ```
- Verify the generated model files in the `models/` directory:
  - `churn_model.pkl` - Trained Random Forest model
  - `preprocessing.pkl` - Label encoders and scaler for data preprocessing
  - `metrics.pkl` - Model evaluation metrics
- Review the model performance metrics (accuracy, precision, recall, F1-score)
- Document the model training process and results

**What the Training Script Does:**
1. Loads the customer churn dataset
2. Preprocesses data (encodes categorical variables, scales numerical features)
3. Splits data into train/test sets (80/20)
4. Trains a Random Forest Classifier with optimal hyperparameters
5. Evaluates model performance on test data
6. Saves model, preprocessing objects, and metrics as pickle files

**Best Practices:**
- Keep the `models/` directory in your repository for deployment
- Document model performance metrics in README
- Include all dependencies in `requirements.txt`
- Test model loading and prediction locally before deployment

---

### 2. Model Deployment (Primary Focus)

**Scenario:**  
Deploy the selected trained model as a production-ready **FastAPI** REST API. Use **Docker** for containerization and deploy using **EITHER Kubernetes OR Docker Compose** (choose one based on your preference and experience). The deployment should follow MLOps best practices.

#### 2.1 Containerization with Docker

**Task:**
- Create a **Dockerfile** containing:
  - The model and dependencies
  - All required libraries
  - FastAPI application code
- Build and test the Docker image locally
- Configure environment variables

#### 2.2 FastAPI Development

**Task:**
- Develop a RESTful API using **FastAPI**:
  - `/predict` endpoint for single predictions
  - `/batch-predict` endpoint for batch predictions
  - `/health` endpoint for health checks
  - `/readiness` endpoint for readiness probes
- Implement input validation using Pydantic models
- Add comprehensive error handling and logging

**Best Practices:**
- Use automatic OpenAPI/Swagger documentation
- Implement asynchronous endpoints for better performance
- Implement middleware for CORS, logging, and error handling

#### 2.3 Deployment: Choose ONE Option

**You must choose EITHER Option A (Kubernetes) OR Option B (Docker Compose) - NOT BOTH**

##### Option A: Kubernetes Deployment

**Task:**
- Create basic Kubernetes manifests:
  - **Deployment**: Define replicas and resource limits
  - **Service**: Expose the API
  - **ConfigMap**: Store configuration parameters
- Set up readiness and liveness probes
- Deploy to a Kubernetes cluster (local minikube/kind or cloud AKS/EKS/GKE)

##### Option B: Docker Compose Deployment

**Task:**
- Create a `docker-compose.yml` file:
  - Define the FastAPI service
  - Configure environment variables
  - Set up port mappings
  - Configure health checks

---

### 3. Version Control and Repository Structure

**Task:**
- Create a Git repository with the following structure:
  ```
  ├── src/                     # FastAPI source code
  │   ├── __init__.py
  │   ├── main.py             # FastAPI application
  │   └── models.py           # Pydantic models for validation
  ├── models/                  # Trained model files (generated from train_model.py)
  │   ├── churn_model.pkl
  │   ├── preprocessing.pkl
  │   └── metrics.pkl
  ├── tests/                   # Unit and integration tests
  │   ├── __init__.py
  │   ├── test_api.py
  │   └── test_prediction.py
  ├── config/                  # Configuration files
  ├── train_model.py           # Model training script (provided)
  ├── customer_churn_dataset.csv  # Training dataset (provided)
  ├── Dockerfile               # Docker configuration
  ├── docker-compose.yml       # If using Docker Compose
  ├── kubernetes/              # If using Kubernetes (manifests)
  │   ├── deployment.yaml
  │   ├── service.yaml
  │   └── configmap.yaml
  ├── requirements.txt         # Python dependencies
  ├── README.md                # Project documentation
  └── CICD_APPROACH.md        # CI/CD pipeline approach document
  ```
- Use feature branches and pull requests for development
- Write clear commit messages
- Keep the repository organized and clean

**Best Practices:**
- Use `.gitignore` to exclude virtual environments, `__pycache__`, etc.
- Don't commit large model files if using Git (document where to generate them)
- Use meaningful branch names (feature/api-endpoints, fix/validation-bug)
- Tag releases with semantic versioning

---

### 4. Testing

**Task:**
- Implement basic automated testing:
  - **Unit tests**: Test individual components (model loading, preprocessing, prediction logic)
  - **Integration tests**: Test API endpoints
  - Use pytest framework
  - Achieve reasonable test coverage

**Best Practices:**
- Test model loading and prediction functions
- Test API endpoints with various inputs (valid, invalid, edge cases)
- Test error handling
- Keep tests simple and maintainable

---

## Deliverables

### 1. Codebase
- **Git repository** containing **all necessary working files and scripts** to replicate and run the solution locally:
  - Training script (`train_model.py`) and dataset (`customer_churn_dataset.csv`)
  - Trained model files in `models/` directory (generated from training)
  - Complete FastAPI implementation with all endpoints
  - Dockerfile (tested and working)
  - **EITHER** Kubernetes manifests (Deployment, Service, ConfigMap) **OR** docker-compose.yml
  - Automated tests (unit and integration tests)
  - All configuration files needed to run the application
  - requirements.txt with all dependencies
  - Any helper scripts or utilities used

**Important**: The repository should be fully functional and replicable. A reviewer should be able to:
- Clone the repository
- Follow the README instructions
- Train the model
- Run the application locally (with Docker or directly)
- Execute tests
- Deploy using the provided manifests

### 2. Documentation
- **README.md** with clear instructions:
  - Prerequisites and setup instructions
  - How to train the model and view results
  - How to run the application (locally and with Docker)
  - Deployment instructions
  - API usage examples
  - Evidence of working deployment (screenshots or video)

### 3. Bonus (Optional)
- **CI/CD Approach**: Document your proposed approach for implementing a CI/CD pipeline (1-2 pages in CICD_APPROACH.md or as a section in README)

---

## Timeline Recommendation

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| **Phase 1: Setup & Model Training** | 1 day | Repository setup, environment setup, run train_model.py, verify model artifacts |
| **Phase 2: FastAPI Development** | 2 days | FastAPI implementation, load model, Pydantic models, input validation, local testing |
| **Phase 3: Dockerization** | 1 day | Create Dockerfile, build image, test container locally |
| **Phase 4: Deployment Setup** | 1-2 days | Create Kubernetes manifests OR docker-compose.yml, deploy and test |
| **Phase 5: Testing & Documentation** | 1-2 days | Write tests, README, API documentation |

**Total Estimated Time: 5-7 days**

---

## Getting Started

### Prerequisites
- Docker Desktop installed
- Minikube/kind installed (only if using Kubernetes, otherwise Docker setup would cover docker-compose)
- Python 3.9+ installed
- Git installed

### Quick Start Guide

1. **Repository Setup**
   ```bash
   git clone <repository-url>
   cd mlops-churn-prediction
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   # Run the training script
   python train_model.py
   
   # Verify model files are generated in models/ directory
   ls models/  # Should see: churn_model.pkl, preprocessing.pkl, metrics.pkl
   ```

3. **Develop FastAPI Application**
   ```bash
   # Create src/main.py with FastAPI endpoints
   # Load the trained model and preprocessing objects
   # Implement prediction endpoints
   
   # Run FastAPI locally
   uvicorn src.main:app --reload
   
   # Test the API
   curl http://localhost:8000/health
   ```

4. **Containerize Application**
   ```bash
   # Build Docker image
   docker build -t churn-api:latest .
   
   # Run container locally
   docker run -p 8000:8000 churn-api:latest
   
   # Test containerized API
   curl http://localhost:8000/health
   ```

5. **Deploy Application**
   
   **Option A - Using Docker Compose:**
   ```bash
   docker-compose up -d
   docker-compose ps
   ```
   
   **Option B - Using Kubernetes:**
   ```bash
   kubectl apply -f kubernetes/
   kubectl get pods
   ```

6. **Run Tests**
   ```bash
   # Run unit and integration tests
   pytest tests/ -v
   
   # Run with coverage
   pytest tests/ --cov=src --cov-report=html
   ```

7. **Validate Deployment**
   ```bash
   # Test API endpoint
   curl -X POST http://<endpoint>/v1/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 45, "tenure_months": 24, "monthly_charges": 79.85, ...}'
   ```

8. **Document CI/CD Approach**
   - Create `CICD_APPROACH.md`
   - Document your pipeline design and reasoning
