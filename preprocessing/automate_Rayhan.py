"""
Automated preprocessing script for Telco Customer Churn dataset.

Converts manual experimentation steps from Eksperimen_Nama-siswa.ipynb
into reusable functions that produce ready-to-train data.

Usage:
    python automate_Nama-siswa.py

Output:
    telco_churn_preprocessing/X_train.csv
    telco_churn_preprocessing/X_test.csv
    telco_churn_preprocessing/y_train.csv
    telco_churn_preprocessing/y_test.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# Configuration — paths resolved relative to this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # parent of preprocessing/
INPUT_CSV = os.path.join(PROJECT_DIR, "telco_churn_raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "telco_churn_preprocessing")
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and return DataFrame."""
    print(f"[1/8] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Shape: {df.shape}  |  Columns: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Drop identifier column
# ---------------------------------------------------------------------------
def drop_identifier(df: pd.DataFrame, col: str = "customerID") -> pd.DataFrame:
    """Drop the identifier column (not a feature)."""
    print(f"[2/8] Dropping identifier column: '{col}'")
    return df.drop(col, axis=1)


# ---------------------------------------------------------------------------
# Step 3: Clean TotalCharges column
# ---------------------------------------------------------------------------
def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges from object to float and fill missing values."""
    print("[3/8] Cleaning TotalCharges column...")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"      NaN values filled: {nan_count} (tenure=0 new customers)")
    return df


# ---------------------------------------------------------------------------
# Step 4: Encode target variable
# ---------------------------------------------------------------------------
def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode target column (Yes/No → 1/0)."""
    print(f"[4/8] Encoding target: '{target_col}'")
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"      Mapping: {mapping}")
    return df, le


# ---------------------------------------------------------------------------
# Step 5: Encode binary categorical features
# ---------------------------------------------------------------------------
def encode_binary_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode binary categorical columns."""
    print("[5/8] Encoding binary categorical features...")
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"      {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df, encoders


# ---------------------------------------------------------------------------
# Step 6: One-Hot Encode multi-class categorical features
# ---------------------------------------------------------------------------
def onehot_encode(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode multi-class categorical columns (drop='first')."""
    print("[6/8] One-Hot Encoding multi-class categorical features...")
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
    ]

    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    ohe_array = ohe.fit_transform(X[multi_cols])
    ohe_names = ohe.get_feature_names_out(multi_cols)

    X_ohe = pd.DataFrame(ohe_array, columns=ohe_names, index=X.index)
    X = X.drop(multi_cols, axis=1)
    X = pd.concat([X, X_ohe], axis=1)

    print(f"      Added {len(ohe_names)} one-hot columns (drop='first')")
    return X, ohe


# ---------------------------------------------------------------------------
# Step 7: Scale numeric features
# ---------------------------------------------------------------------------
def scale_numeric(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler to numeric columns."""
    print("[7/8] Scaling numeric features with StandardScaler...")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print(f"      Scaled: {numeric_cols}")
    return X, scaler


# ---------------------------------------------------------------------------
# Step 8: Train/Test split and save
# ---------------------------------------------------------------------------
def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> None:
    """Stratified train/test split and save to CSV."""
    print("[8/8] Performing stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"[DONE] Preprocessed data saved to: {output_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"\n  y_train dist: {y_train.value_counts().to_dict()}")
    print(f"  y_test dist:  {y_test.value_counts().to_dict()}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Orchestrator — run full preprocessing pipeline
# ---------------------------------------------------------------------------
def run_preprocessing(
    input_path: str = INPUT_CSV,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Run the complete preprocessing pipeline end-to-end."""
    print("=" * 60)
    print("  AUTOMATED PREPROCESSING — Telco Customer Churn")
    print("=" * 60)

    # 1. Load
    df = load_data(input_path)

    # 2. Drop ID
    df = drop_identifier(df)

    # 3. Clean TotalCharges
    df = clean_total_charges(df)

    # 4. Encode target
    df, target_encoder = encode_target(df)

    # Separate features and target
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # 5. Encode binary categorical
    X, binary_encoders = encode_binary_features(X)

    # 6. One-hot encode multi-class
    X, ohe = onehot_encode(X)

    # 7. Scale numeric
    X, scaler = scale_numeric(X)

    # 8. Split and save
    split_and_save(X, y, output_dir)

    print(f"\n[INFO] Feature dimensions after preprocessing: {X.shape[1]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
