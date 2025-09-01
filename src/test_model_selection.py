import pandas as pd
import os
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Define model and metadata save directory
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_featured_data():
    """Load feature-engineered dataset."""
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    input_path = os.path.join(PROCESSED_DIR, "train_fe.csv")
    df = pd.read_csv(input_path)
    return df


def prepare_data(df, target="is_fraud"):
    """Split features and target."""
    X = df.drop(columns=[
        target, "cc_num", "trans_num", "first", "last", "street",
        "city", "state", "dob", "trans_date_trans_time"
    ])
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size=0.3, random_state=42):
    """Split into train and test sets using stratified sampling."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_model_catalog(scale_pos_weight=1):
    """Return a catalog of preconfigured models."""
    models = {
        "logistic_regression (low precision, average recall)": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver='lbfgs'
        ),

        "random_forest (low precision, high recall)": RandomForestClassifier(
        n_estimators=30,
        class_weight='balanced',
        max_depth=7,
        random_state=42,
        n_jobs=-1
        ),
        "xgboost (balanced above 70)": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=42
        ),
        "xgboost_TP (high recall, low precision, catch max frauds)": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=250,
            eval_metric='auc',
            random_state=42
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print classification metrics."""
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print("\n🧮 Confusion Matrix:")
    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": acc,
        "confusion_matrix": cm
    }


def save_model(model, model_name, features_used):
    """
    Save trained model and its metadata.
    Model saved as: models/<model_name>.joblib
    Metadata saved as: models/<model_name>_meta.json
    """
    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}_meta.json"

    # Save the model
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Save model metadata (e.g., features used)
    metadata = {
        "model_name": model_name,
        "features_used": features_used
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"📝 Metadata saved: {metadata_path}\n")


def select_best_model(X_train, y_train, X_test, y_test, top_features, scoring="f1", scale_pos_weight = 5):
    """
    Train all models in the catalog and select the best one.
    Save each trained model and their top features.
    """

    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    catalog = get_model_catalog(scale_pos_weight=scale_pos_weight)

    results = {}

    print("\n🔍 Evaluating all models:\n")

    for name, model in catalog.items():
        print(f"➡️  Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Save each model and features used
        save_model(model, name, top_features)

    best_model_name = max(results, key=lambda name: results[name][scoring])
    best_model = results[best_model_name]["model"]

    print(f"\n✅ Best model: {best_model_name} (based on {scoring})")

    return best_model_name, best_model, results[best_model_name]


def select_top_features_xgboost(X_train, y_train, top_n=8, scale_pos_weight=1):
    """
    Train XGBoost model and select top N features based on feature importance.
    Returns the list of top feature names.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Get feature importances and sort
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance_df.head(top_n)['feature'].tolist()
    print(f"\n🏆 Top {top_n} features selected by XGBoost:")
    print(top_features)
    return top_features


# Example usage for testing
if __name__ == "__main__":
    df = load_featured_data()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = 5

    # Select top N features
    top_features = select_top_features_xgboost(X_train, y_train, top_n=10, scale_pos_weight=scale_pos_weight)

    # Use only top features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # Train all models, save each, and pick the best
    best_name, best_model, metrics = select_best_model(
        X_train_top, y_train,
        X_test_top, y_test,
        top_features=top_features,
        scoring="f1", scale_pos_weight=scale_pos_weight)

    print("\n📌 Final Evaluation of Best Model:")
    evaluate_model(best_model, X_test_top, y_test)

    print("\n✅ Training complete.")
