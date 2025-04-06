import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictStr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import time
import logging
import uuid
from datetime import datetime

os.makedirs("logs", exist_ok=True) # Make sure directory exits to prevent crash

# Structured logs to be saved
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

# --------------------
# Load Model and Tokenizer
# --------------------

model_path = "MiladMt/sentiment-api-model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# --------------------
# Set Up FastAPI App
# --------------------

app = FastAPI()

# Request body schema
class ReviewInput(BaseModel):
    text: StrictStr

# ID to label mapping {0: "negative", 1: "neutral", 2: "positive"}
id2label = model.config.id2label

# Get the number of max tokens accepted by the model
MAX_TOKENS = tokenizer.model_max_length

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

    # Reject empty string input
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text must not be empty")
    
    # Truncatiion check for inputs longer than MAX_TOKENS accepted by the model
    tokenized_untruncated = tokenizer(input_text, return_tensors="pt", truncation=False)
    token_count = tokenized_untruncated["input_ids"].shape[1]

    was_truncated = token_count > MAX_TOKENS

    # Tokenize input for inference
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
    f"Input text:  {input_text}\n"
    f"Input length:  {input_length}\n"
    f"Prediction:    {prediction_label}\n"
    f"Input truncated:     {was_truncated}\n"
    f"Latency:       {latency_ms}ms\n"
    )


    return {
        "prediction": prediction_label,
        "latency_ms": latency_ms,
        "input_truncated": was_truncated
    }
