import pickle
import os


def load_metrics():
    metrics_path = os.path.join("models", "metrics.pkl")

    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run train_model.py first.")
        return

    print(f"Loading metrics from {metrics_path}...")
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    print("\n--- Model Performance Report ---")
    # metrics is a dictionary based on typical patterns
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print(metrics)
    print("--------------------------------")


if __name__ == "__main__":
    load_metrics()
