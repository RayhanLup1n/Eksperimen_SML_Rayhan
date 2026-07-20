import json
import os
import urllib.request

import pandas as pd


BASE_DIR = r"B:\project\digdaya\practitioner\Membangung_Sistem_Machine_Learning"
PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "Membangun_Model_New",
    "telco_churn_preprocessing",
    "telco_churn_preprocessed.csv"
)

def build_payload(path: str) -> dict:
    """
    Build inference payload from the first row of preprocessed data.
    MLflow scoring server expects dataframe_split format.
    """
    df = pd.read_csv(path)
    features = df.drop(columns=["Churn"])
    sample = features.iloc[[0]]

    return {
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist()
        }
    }

def predict() -> dict:
    """
    Send a sample prediction request to the served model.
    """
    payload = build_payload(PREPROCESSED_PATH)
    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:5002/invocations",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    result = predict()
    print("Prediction result:", result)