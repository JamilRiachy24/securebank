
import json
import joblib
import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
EVAL_LOG_PATH = LOGS_DIR / "model_evaluation_log.json"


def list_saved_models():
    """List all saved models and their metadata."""
    models = []
    for file in MODEL_DIR.glob("*_meta.json"):
        model_name = file.stem.replace("_meta", "")
        metadata_path = MODEL_DIR / f"{model_name}_meta.json"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        models.append({
            "model_name": model_name,
            "features_used": metadata.get("features_used", [])
        })
    return models


def load_saved_model(model_name):
    """Load the saved model and associated metadata."""
    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}_meta.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found in models directory.")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


def load_test_data(features_used):
    """Load test data and return X_test and y_test with only selected features."""
    test_path = PROCESSED_DATA_DIR / "train_fe.csv"
    df = pd.read_csv(test_path)

    X = df[features_used]
    y = df["is_fraud"]

    return X, y


def evaluate(model, X_test, y_test):
    """Print classification report and return metrics."""
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": acc,
        "confusion_matrix": cm.tolist()  # convert numpy array to list for JSON
    }


def save_evaluation_log(model_name, features, metrics):
    """Append evaluation result to the JSON log file."""
    log_entry = {
        "model_name": model_name,
        "features_used": features,
        "evaluated_at": datetime.datetime.now().isoformat(),
        "metrics": metrics
    }

    if EVAL_LOG_PATH.exists():
        with open(EVAL_LOG_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(EVAL_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"\n📝 Evaluation results saved to: {EVAL_LOG_PATH}")


def admin_model_selection_workflow():
    """Run full admin selection process."""
    print("🔎 Scanning for saved models...\n")

    models = list_saved_models()

    if not models:
        print("❌ No saved models found.")
        return

    print("📋 Available models:\n")
    for idx, m in enumerate(models):
        print(f"{idx + 1}. {m['model_name']}")
        print(f"   Features used: {m['features_used']}\n")

    while True:
        try:
            choice = int(input("🔧 Enter model number to load (0 to cancel): "))
            if choice == 0:
                print("🚪 Exiting.")
                return
            selected_model = models[choice - 1]
            break
        except (ValueError, IndexError):
            print("❗ Invalid choice. Try again.")

    model_name = selected_model["model_name"]
    features_used = selected_model["features_used"]

    model, metadata = load_saved_model(model_name)
    print(f"\n✅ Loaded model: {model_name}")
    print(f"📌 Features used: {features_used}")

    X_test, y_test = load_test_data(features_used)
    metrics = evaluate(model, X_test, y_test)

    save_evaluation_log(model_name, features_used, metrics)


if __name__ == "__main__":
    admin_model_selection_workflow()
