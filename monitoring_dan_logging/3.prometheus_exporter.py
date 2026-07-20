import json
import os
import time
import urllib.request

import pandas as pd
from prometheus_client import Counter, Histogram, start_http_server

# Path configuration
BASE_DIR = r"B:\project\digdaya\practitioner\Membangung_Sistem_Machine_Learning"
PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "Membangun_Model_New",
    "telco_churn_preprocessing",
    "telco_churn_preprocessed.csv"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "model_inference_requests_total",
    "Total number of inference requests",
    ["status"]
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time spent on inference request"
)

def load_sample_payload(path: str) -> dict:
    """
    Load a single preprocessed row and format it for MLflow serving.
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

def make_inference_request() -> dict:
    """
    Send a sample inference request to the served model.
    Returns the prediction response.
    """
    payload = load_sample_payload(PREPROCESSED_PATH)
    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:5002/invocations",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start_time = time.time()
    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
            REQUEST_COUNT.labels(status="success").inc()
            return result
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise e
    finally:
        INFERENCE_LATENCY.observe(time.time() - start_time)

def main() -> None:
    """
    Start Prometheus exporter and periodically send inference requests.
    """
    start_http_server(8000)
    print("Prometheus exporter started on http://127.0.0.1:8000")

    while True:
        try:
            result = make_inference_request()
            print(f"Inference result: {result}")
        except Exception as e:
            print(f"Inference error: {e}")
        time.sleep(5)

if __name__ == "__main__":
    main()