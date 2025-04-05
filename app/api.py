from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import time
import logging
import uuid
from datetime import datetime

# Structured logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)



# --------------------
# Load Model and Tokenizer
# --------------------

model_path = "./model_3epochs"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# --------------------
# Set Up FastAPI App
# --------------------

app = FastAPI()

# Request body schema
class ReviewInput(BaseModel):
    text: str

# ID to label mapping {0: "negative", 1: "neutral", 2: "positive"}
id2label = model.config.id2label

# --------------------
# Predict Endpoint
# --------------------

@app.post("/predict")
async def predict_sentiment(review: ReviewInput):
    request_id = str(uuid.uuid4())  # Unique ID per request
    timestamp = datetime.utcnow().isoformat()
    start_time = time.time()

    input_text = review.text
    input_length = len(input_text)

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()
        prediction_label = id2label[prediction_id]

    latency_ms = round((time.time() - start_time) * 1000, 2)  # In milliseconds

    # Structured log message
    logging.info(
    f"\n[Request ID: {request_id[:8]}]\n"
    f"Timestamp:     {timestamp}\n"
    f"Input length:  {input_length}\n"
    f"Prediction:    {prediction_label}\n"
    f"Latency:       {latency_ms}ms\n"
    )


    return {
        "prediction": prediction_label,
        "latency_ms": latency_ms
    }
