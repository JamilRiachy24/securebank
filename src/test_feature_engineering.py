# securebank/src/feature_engineering.py

import os
import pandas as pd


def add_time_features(df):
    """Extract hour, day of week, and month from transaction datetime."""
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # Monday=0, Sunday=6
    df["month"] = df["trans_date_trans_time"].dt.month
    return df


def add_fraud_cluster_features(df):
    """
    Add features capturing fraud clustering:
    - Time since previous transaction (per customer)
    - Number of transactions in past hour (per customer)
    """

    # Sort by customer and time
    df = df.sort_values(by=["cc_num", "trans_date_trans_time"]).copy()

    # Time since previous transaction per customer (in seconds)
    df["time_since_prev"] = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()

    # Calculate rolling count per customer using rolling on datetime column
    transactions_past_hour = pd.Series(index=df.index, dtype=float)

    for cc_num, group in df.groupby("cc_num"):
        # Use rolling with 'on' parameter to keep the original index
        rolling_counts = group.rolling('1h', on='trans_date_trans_time')['trans_num'].count()
        transactions_past_hour.loc[group.index] = rolling_counts.values

    df["transactions_past_hour"] = transactions_past_hour

    # Fill NaNs for first transactions
    df["time_since_prev"] = df["time_since_prev"].fillna(999999)
    df["transactions_past_hour"] = df["transactions_past_hour"].fillna(1)

    return df



def add_customer_age(df):
    """Calculate customer age at transaction time."""
    df["dob"] = pd.to_datetime(df["dob"], format='mixed', dayfirst=True, errors='coerce')
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    return df

def label_encode_columns(df):
    """Convert categorical columns into numerical labels (label encoding)."""
    from sklearn.preprocessing import LabelEncoder

    columns_to_encode = ["sex", "merchant", "category", "job", "merch_lat"]

    for col in columns_to_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings
        else:
            print(f"Warning: Column '{col}' not found in dataframe.")

    return df



def feature_engineering_pipeline(df):
    """Run all feature engineering steps."""
    df = add_time_features(df)
    df = add_fraud_cluster_features(df)
    df = add_customer_age(df)
    df = label_encode_columns(df)

    df = df.dropna(axis=0)

    return df



# Example usage:
# Example usage:
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    input_path = os.path.join(PROCESSED_DIR, "train.csv")

    print("Loading preprocessed data...")
    df = pd.read_csv(input_path, parse_dates=["trans_date_trans_time", "dob"])

    print("Engineering features...")
    df_fe = feature_engineering_pipeline(df)

    output_path = os.path.join(PROCESSED_DIR, "train_fe.csv")
    df_fe.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to {output_path}")
