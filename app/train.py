import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)



# ----------------------
# Load and Prepare Data
# ----------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from CSV.

    Args:
        path (str): Path to the processed CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
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
    Converts a pandas DataFrame to a HuggingFace Dataset format,
    keeping only the required columns.

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

def main():
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
        output_dir="./outputs",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    # ----------------------
    # Trainer Setup and Training Execution
    # ----------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Save the final model
    trainer.save_model("./model")
    


if __name__ == "__main__":
    main()
