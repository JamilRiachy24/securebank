# securebank/main.py

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Imports from the pipeline
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.model_selection import (
    load_featured_data, prepare_data, train_test_split_data,
    select_top_features_xgboost, select_best_model, evaluate_model
)
from src.model_training import admin_model_selection_workflow
from src.audit import load_evaluation_logs, summarize_metrics


def run_full_pipeline():
    print("\n🚀 Starting SecureBank Fraud Detection Pipeline\n")

    # Step 1: Preprocessing
    print("🔄 Step 1: Data Preprocessing")
    df_clean = preprocess_pipeline(save=True)

    # Step 2: Feature Engineering
    print("\n🔧 Step 2: Feature Engineering")
    df_fe = feature_engineering_pipeline(df_clean)

    # Save engineered data
    processed_dir = BASE_DIR / "processed_data"
    output_path = processed_dir / "train_fe.csv"
    df_fe.to_csv(output_path, index=False)
    print(f"✅ Feature-engineered data saved to {output_path}")

    # Step 3: Load data and prepare
    print("\n📊 Step 3: Model Training")
    df = load_featured_data()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Step 4: Feature Selection
    scale_pos_weight = 5
    top_features = select_top_features_xgboost(X_train, y_train, top_n=10, scale_pos_weight=scale_pos_weight)

    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    # Step 5: Train and Evaluate Models
    best_model_name, best_model, _ = select_best_model(
        X_train_top, y_train,
        X_test_top, y_test,
        top_features=top_features,
        scoring="f1",
        scale_pos_weight=scale_pos_weight
    )

    print("\n🎯 Step 6: Final Evaluation of Best Model")
    evaluate_model(best_model, X_test_top, y_test)

    # Step 7: Admin model selection (interactive)
    print("\n🛠️ Step 7: Admin Model Review (Interactive)")
    admin_model_selection_workflow()

    # Step 8: Summarize Logs
    print("\n📜 Step 8: Evaluation Log Summary")
    logs = load_evaluation_logs()
    summarize_metrics(logs)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    run_full_pipeline()
