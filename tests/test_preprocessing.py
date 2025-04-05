import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from app.train import map_sentiment_labels


def test_map_sentiment_labels_basic():
    """
    Should correctly map known sentiment values to numeric labels.
    """
    df = pd.DataFrame({"sentiment": ["positive", "neutral", "negative"]})
    result = map_sentiment_labels(df)

    assert list(result["label"]) == [2, 1, 0]


def test_map_sentiment_labels_handles_unknown():
    """
    Should return NaN for unknown sentiment values.
    """
    df = pd.DataFrame({"sentiment": ["positive", "invalid", "negative"]})
    result = map_sentiment_labels(df)

    assert result.loc[0, "label"] == 2
    assert pd.isna(result.loc[1, "label"])
    assert result.loc[2, "label"] == 0


def test_map_sentiment_labels_with_nulls():
    """
    Should return NaN when sentiment is None or missing.
    """
    df = pd.DataFrame({"sentiment": ["neutral", None, "negative"]})
    result = map_sentiment_labels(df)

    assert result.loc[0, "label"] == 1
    assert pd.isna(result.loc[1, "label"])
    assert result.loc[2, "label"] == 0
