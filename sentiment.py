"""
sentiment.py
------------
Score earnings call transcripts using FinBERT.
"""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm


def load_finbert():
    """Load the FinBERT sentiment pipeline."""
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
    )


def score_transcript(text: str, pipe, chunk_size: int = 400, max_chunks: int = 10) -> float:
    """
    Score a single transcript.

    FinBERT has a 512-token limit, so we chunk the text and average
    across chunks. Returns a net sentiment score from -1 to +1.
    """
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [c for c in chunks if c.strip()][:max_chunks]

    if not chunks:
        return 0.0

    scores = pipe(chunks)
    pos = sum(1 for s in scores if s["label"] == "positive")
    neg = sum(1 for s in scores if s["label"] == "negative")
    return (pos - neg) / len(scores)


def score_all(df: pd.DataFrame, pipe, text_col: str = "transcript") -> pd.DataFrame:
    """
    Add a 'sentiment' column to a DataFrame of transcripts.

    Args:
        df:       DataFrame with at least a text column and a 'ticker' column.
        pipe:     FinBERT pipeline from load_finbert().
        text_col: Name of the column containing transcript text.
    """
    tqdm.pandas(desc="Scoring transcripts")
    df = df.copy()
    df["sentiment"] = df[text_col].progress_apply(
        lambda t: score_transcript(t, pipe)
    )
    return df
