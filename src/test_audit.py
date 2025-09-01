import json
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path(__file__).parent.parent / "logs"
EVAL_LOG_PATH = LOGS_DIR / "model_evaluation_log.json"


def load_evaluation_logs():
    """Load evaluation logs from the JSON file."""
    if not EVAL_LOG_PATH.exists():
        print(f"❌ Evaluation log file not found at {EVAL_LOG_PATH}")
        return []

    with open(EVAL_LOG_PATH, "r") as f:
        logs = json.load(f)
    return logs


def summarize_metrics(logs):
    """Summarize evaluation metrics for all logged models."""
    if not logs:
        print("⚠️ No evaluation logs to summarize.")
        return

    print(f"\n📈 Summary of {len(logs)} evaluation entries:\n")
    for entry in logs:
        model_name = entry.get("model_name", "Unknown")
        eval_time = entry.get("evaluated_at", "Unknown time")
        metrics = entry.get("metrics", {})

        precision = metrics.get("precision", None)
        recall = metrics.get("recall", None)
        f1 = metrics.get("f1_score", None)
        accuracy = metrics.get("accuracy", None)

        print(f"Model: {model_name}")
        print(f"Evaluated At: {eval_time}")
        print(f" Precision: {precision:.4f}" if precision is not None else " Precision: N/A")
        print(f" Recall:    {recall:.4f}" if recall is not None else " Recall: N/A")
        print(f" F1 Score:  {f1:.4f}" if f1 is not None else " F1 Score: N/A")
        print(f" Accuracy:  {accuracy:.4f}" if accuracy is not None else " Accuracy: N/A")
        print("-" * 30)


def filter_logs_by_model(logs, model_name):
    """Filter logs to include only entries for the specified model."""
    filtered = [log for log in logs if log.get("model_name") == model_name]
    return filtered


def filter_logs_by_date_range(logs, start_date=None, end_date=None):
    """
    Filter logs by evaluation date range.
    Dates should be datetime.date or datetime.datetime objects.
    """
    filtered = []
    for log in logs:
        eval_time_str = log.get("evaluated_at")
        if not eval_time_str:
            continue
        eval_time = datetime.fromisoformat(eval_time_str)

        if start_date and eval_time < start_date:
            continue
        if end_date and eval_time > end_date:
            continue

        filtered.append(log)
    return filtered


def main():
    print("🔎 Loading evaluation logs...\n")
    logs = load_evaluation_logs()
    if not logs:
        return

    # Simple summary of all logs
    summarize_metrics(logs)

    # Optional: let admin filter logs interactively
    while True:
        user_input = input("\nWould you like to filter logs by model name? (y/n): ").strip().lower()
        if user_input == "y":
            model_name = input("Enter exact model name to filter: ").strip()
            filtered_logs = filter_logs_by_model(logs, model_name)
            if filtered_logs:
                print(f"\nFound {len(filtered_logs)} entries for model '{model_name}':\n")
                summarize_metrics(filtered_logs)
            else:
                print(f"No logs found for model '{model_name}'.")
        else:
            break

    print("\n✅ Audit complete.")


if __name__ == "__main__":
    main()
