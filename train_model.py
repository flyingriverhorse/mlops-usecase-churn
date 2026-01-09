"""
Model Training Script for Customer Churn Prediction
This script trains a Random Forest model on the customer churn dataset
and saves the trained model and preprocessing objects as pickle files.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Load the customer churn dataset"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['churn'].value_counts()}")
    return df


def preprocess_data(df):
    """Preprocess the data: encode categorical variables and scale numerical features"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Store preprocessing objects
    preprocessing = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_columns': X.columns.tolist()
    }
    
    return X, y, preprocessing


def train_model(X_train, y_train):
    """Train a Random Forest Classifier"""
    print("\nTraining Random Forest model...")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_artifacts(model, preprocessing, metrics, model_path='models/churn_model.pkl', 
                   preprocessing_path='models/preprocessing.pkl', metrics_path='models/metrics.pkl'):
    """Save model, preprocessing objects, and metrics"""
    print("\nSaving model artifacts...")
    
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save preprocessing objects
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing, f)
    print(f"Preprocessing objects saved to {preprocessing_path}")
    
    # Save metrics
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("Customer Churn Prediction - Model Training")
    print("="*60)
    
    # Load data
    df = load_data('customer_churn_dataset.csv')
    
    # Preprocess data
    X, y, preprocessing = preprocess_data(df)
    
    # Split data
    print("\nSplitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    save_artifacts(model, preprocessing, metrics)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - models/churn_model.pkl        (trained model)")
    print("  - models/preprocessing.pkl      (encoders and scaler)")
    print("  - models/metrics.pkl            (evaluation metrics)")
    print("\nYou can now use these files for deployment in your FastAPI application.")


if __name__ == "__main__":
    main()
