import re
import pandas as pd

def map_sentiment(rating: int) -> str:
    """
    Map numeric rating to sentiment label.

    1–2 => 'negative'
    3   => 'neutral'
    4–5 => 'positive'

    Args:
        rating (int): Original star rating

    Returns:
        str: Sentiment label
    """
    if rating in [1, 2]:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

def clean_text(text: str) -> str:
    """
    Minimal text cleaning suitable for transformer models.

    - Strips leading/trailing whitespace
    - Normalizes multiple spaces

    Args:
        text (str): Raw review text

    Returns:
        str: Cleaned review text
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment mapping and text cleaning to a DataFrame.

    Keeps only the necessary columns and returns a processed DataFrame
    with 'text_clean' and 'sentiment'.

    Args:
        df (pd.DataFrame): Raw DataFrame with 'text' and 'rating'

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'text_clean' and 'sentiment'
    """
    df = df[["rating", "text"]].copy()
    df["sentiment"] = df["rating"].apply(map_sentiment)
    df["text_clean"] = df["text"].apply(clean_text)
    return df[["text_clean", "sentiment"]]
