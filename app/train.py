import argparse
import pandas as pd
from datasets import Dataset
import mlflow
from transformers.integrations import MLflowCallback
import mlflow.transformers

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
)



# ----------------------
# Load and Prepare Data
# ----------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from CSV.
    """
    return pd.read_csv(path)


def map_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map sentiment labels to numeric IDs for model training.

    Args:
        df (pd.DataFrame): DataFrame with a 'sentiment' column.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'label' column.
    """
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label2id)
    return df



# ----------------------
# Dataset Conversion & Tokenization
# ----------------------

def convert_to_hf_dataset(df: pd.DataFrame, text_column: str = "text_clean") -> Dataset:
    """
    Converts a pandas DataFrame to a HuggingFace Dataset format.

    Args:
        df (pd.DataFrame): DataFrame with text and label columns.
        text_column (str): The name of the text column to keep.

    Returns:
        Dataset: HuggingFace Dataset object.
    """
    return Dataset.from_pandas(df[[text_column, "label"]])


def tokenize_dataset(dataset: Dataset, tokenizer, text_column: str = "text_clean") -> Dataset:
    """
    Tokenizes a HuggingFace Dataset using the provided tokenizer.

    Args:
        dataset (Dataset): Dataset with a 'text_clean' column.
        tokenizer: HuggingFace tokenizer.
        text_column (str): Name of the text column to tokenize.

    Returns:
        Dataset: Tokenized dataset with input_ids and attention_mask.
    """

    def tokenize_batch(batch: dict) -> dict:
        """
        Applies tokenizer to a batch of examples.

        HuggingFace passes a batch (dict of lists), not a single row.
        """
        texts = batch[text_column]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length"
        )

    return dataset.map(tokenize_batch, batched=True)



# ----------------------
# Main
# ----------------------

def main(epochs: int = 3):
    # Define model output paths
    model_dir = f"./model_{epochs}epochs"
    output_dir = f"./outputs_{epochs}epochs"
    artifact_name = f"transformers_model_{epochs}epochs"


    # Load and prepare data
    df = load_data("data/processed_reviews.csv")
    df = map_sentiment_labels(df)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Convert to HF dataset and tokenize
    hf_dataset = convert_to_hf_dataset(df)
    tokenized_dataset = tokenize_dataset(hf_dataset, tokenizer)

    # Split into train/test sets
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ----------------------
    # Model and Training Setup
    # ----------------------

    # Initialize the model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    # ----------------------
    # Set up Trainer with MLflow callback
    # ----------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[MLflowCallback()]
    )

    # MLflow Experiment Tracking
    mlflow.set_experiment("sentiment-classifier")

    with mlflow.start_run(run_name=f"distilbert-{epochs}epochs"):
        # Log training configuration
        mlflow.log_param("model_name", "distilbert-base-uncased")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("eval_size", len(eval_dataset))


        # Train the model
        trainer.train()

        # Evaluate and log metrics
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)
        for key, value in eval_results.items():
            mlflow.log_metric(key, value)

        # Save model and tokenizer
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        # Create a pipeline for MLflow logging
        hf_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer
        )

        # Log to MLflow (Transformers flavor)
        mlflow.transformers.log_model(
            transformers_model=hf_pipeline,
            artifact_path=artifact_name,
            task="text-classification",
        )

    
if __name__ == "__main__":

    # Parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3, help="Number of training epochs")
    args = parser.parse_args()

    main(epochs=args.epochs)
