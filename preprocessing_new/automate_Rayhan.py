import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_dATA_PATH = "telco_churn_raw.csv"
OUTPUT_DIR = "telco_churn_preprocessing"
OUTPUT_FILE = "telco_churn_preprocessed.csv"

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw Telco Customer Churn Dataset from CSV file
    """
    df = pd.read_csv(path)
    print(f'Loaded raw data with shape: {df.shape}')
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and Transfrom raw data into model-ready format.
    Steps:
    -   Drop customerID Column
    -   Convert TotalCharges to numeric
    -   Fill missing TotalCharges with median
    -   Encode target and Categorical Features
    -   Scale numerical features
    """

    # create copy to avoid mutatin original dataframe
    data = df.copy()

    # Drop identified column that is not useful for modeling
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])

    # Convert TotalCharges to numeric, coetcing invalid values to NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges with media value
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Encode binary target variable
    data['Churn'] = data['Churn'].map({
        'No': 0,
        'Yes': 1
    })

    # Identify categorical and numerical columns
    cat_colls = data.select_dtypes(include=['object']).columns.to_list()
    num_colls = data.select_dtypes(include=[np.number]).columns.to_list()
    num_colls.remove('Churn') # Exclude target from scaling

    # Label encode categorical columns
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_colls:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder

    # Standarize numerical columns
    scaler = StandardScaler()
    data[num_colls] = scaler.fit_transform(data[num_colls])

    print(f'Preprocessed data shape: {data.shape}')
    return data

def save_preprocessed_data(df: pd.DataFrame, output_dir:str, output_file:str) -> None:
    """
    Save preprocessed dataframe to CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f'Preprocessed data saved to : {os.path.abspath(output_path)}')

def main() -> None:
    """
    End-to-end preprocessing pipeline
    """
    raw_df = load_raw_data(RAW_dATA_PATH)
    processed_df = preprocess_data(raw_df)
    save_preprocessed_data(processed_df, OUTPUT_DIR, OUTPUT_FILE)

if __name__ == '__main__':
    main()