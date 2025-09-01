# securebank/src/data_preprocessing.py

import pandas as pd
import os
import json

# Define data paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_sources")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

# File paths
CUSTOMER_FILE = os.path.join(DATA_DIR, "customer_release.csv")
TRANSACTION_FILE = os.path.join(DATA_DIR, "transactions_release.parquet")
FRAUD_FILE = os.path.join(DATA_DIR, "fraud_release.json")


def load_data():
    """Load raw datasets from data_sources directory."""
    print("Loading data...")
    customers = pd.read_csv(CUSTOMER_FILE)
    transactions = pd.read_parquet(TRANSACTION_FILE)

    with open(FRAUD_FILE, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        fraud = pd.DataFrame(
            [{"trans_num": k, "is_fraud": v} for k, v in data.items()]
        )
    else:
        fraud = pd.DataFrame(data)

    print(f"Customers: {customers.shape}")
    print(f"Transactions: {transactions.shape}")
    print(f"Fraud Labels: {fraud.shape}")

    return customers, transactions, fraud



def merge_data(customers, transactions, fraud):
    """Merge customer, transaction, and fraud data."""
    print("Merging datasets...")

    # Merge fraud labels onto transactions
    transactions = transactions.merge(fraud, on="trans_num", how="left")

    # Fill missing frauds with 0
    transactions["is_fraud"] = transactions["is_fraud"].fillna(0).astype(int)

    # Merge customer info into transactions
    merged = transactions.merge(customers, on="cc_num", how="left")

    print(f"Merged dataset shape: {merged.shape}")
    return merged


def clean_data(df):
    """Basic data cleaning: remove duplicates, handle missing values, format datetime."""
    print("Cleaning data...")

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Parse datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Optional: Drop or fill missing values
    df = df.dropna(subset=["trans_date_trans_time", "cc_num", "amt"])  # Basic required fields
    print(f"cleaned dataset shape: {df.shape}")
    return df


def save_clean_data(df, filename="train.csv"):
    """Save cleaned data to processed_data directory."""
    output_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


def preprocess_pipeline(save=False):
    """End-to-end preprocessing pipeline."""
    customers, transactions, fraud = load_data()
    merged = merge_data(customers, transactions, fraud)
    clean = clean_data(merged)

    if save:
        save_clean_data(clean)

    return clean


# Optional: Run pipeline when this script is executed directly
if __name__ == "__main__":
    preprocess_pipeline(save=True)
