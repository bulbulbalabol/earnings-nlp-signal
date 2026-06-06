"""
backtest.py
-----------
Build the sentiment signal and run basic return analysis.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def run_ols(df: pd.DataFrame, sentiment_col: str = "sentiment", return_col: str = "return_1d"):
    """
    OLS regression: does sentiment predict next-day return?

    Returns the fitted statsmodels RegressionResults object.
    """
    clean = df[[sentiment_col, return_col]].dropna()
    X = sm.add_constant(clean[sentiment_col])
    model = sm.OLS(clean[return_col], X).fit()
    print(model.summary())
    return model


def quintile_analysis(df: pd.DataFrame, sentiment_col: str = "sentiment", return_col: str = "return_1d"):
    """
    Rank stocks into quintiles by sentiment score and compare forward returns.
    Prints long-short spread and returns a summary DataFrame.
    """
    df = df.copy().dropna(subset=[sentiment_col, return_col])
    df["sentiment_rank"] = df[sentiment_col].rank(pct=True)
    df["quintile"] = pd.qcut(df["sentiment_rank"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    summary = df.groupby("quintile")[return_col].agg(["mean", "std", "count"])
    summary.columns = ["avg_return", "std_return", "n"]
    summary["avg_return_pct"] = summary["avg_return"] * 100

    top = summary.loc["Q5", "avg_return"]
    bot = summary.loc["Q1", "avg_return"]
    print(f"\nLong-short spread (Q5 - Q1): {(top - bot) * 100:.2f}%")
    print(summary.round(4))

    return summary


def plot_quintile_returns(summary: pd.DataFrame, return_horizon: str = "1d", save_path: str = None):
    """Bar chart of average returns by sentiment quintile."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850"]
    ax.bar(summary.index, summary["avg_return_pct"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sentiment Quintile (Q1 = most negative)")
    ax.set_ylabel(f"Avg {return_horizon} Return (%)")
    ax.set_title(f"Forward Returns by FinBERT Sentiment Quintile ({return_horizon})")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_scatter(df: pd.DataFrame, sentiment_col: str = "sentiment", return_col: str = "return_1d", save_path: str = None):
    """Scatter plot of sentiment vs forward return with OLS trendline."""
    clean = df[[sentiment_col, return_col]].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(clean[sentiment_col], clean[return_col] * 100, alpha=0.5, s=20, color="#4C72B0")

    # trendline
    m, b = pd.np.polyfit(clean[sentiment_col], clean[return_col] * 100, 1)
    xs = pd.np.linspace(clean[sentiment_col].min(), clean[sentiment_col].max(), 100)
    ax.plot(xs, m * xs + b, color="#C44E52", linewidth=1.5)

    ax.set_xlabel("FinBERT Sentiment Score")
    ax.set_ylabel(f"{return_col.replace('_', ' ').title()} (%)")
    ax.set_title("Sentiment Score vs Forward Return")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig
