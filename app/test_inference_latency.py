import pandas as pd
import requests
import time
import numpy as np

# ----------------------------
# Load Holdout Data
# ----------------------------

# Load the 10 unseen holdout samples you saved earlier
df = pd.read_json("data/inference_holdout_raw.json", lines=True)

# ----------------------------
# Initialize Metrics
# ----------------------------

latencies = []  # To store round-trip latency for each request

# ----------------------------
# Send Requests to API
# ----------------------------

for i, row in df.iterrows():
    payload = {"text": row["text"]}

    # Measure round-trip request time
    start = time.time()
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    end = time.time()

    # Parse response
    result = response.json()
    latency = (end - start) * 1000  # in milliseconds
    latencies.append(latency)

    # Print individual result
    print(f"[{i+1}] Prediction: {result['prediction']} | API Latency (reported): {result['latency_ms']}ms | Round-trip: {round(latency, 2)}ms")

# Exact p99 measurment
p99 = round(np.percentile(latencies, 99), 2)

# ----------------------------
# Print Summary
# ----------------------------

print("\n-----------------------------")
print(f"Total requests: {len(latencies)}")
print(f"Max round-trip latency: {round(max(latencies), 2)}ms")
print(f"Average round-trip latency: {round(sum(latencies)/len(latencies), 2)}ms")
print(f"P99 round-trip latency: {p99}ms")
print("-----------------------------")
