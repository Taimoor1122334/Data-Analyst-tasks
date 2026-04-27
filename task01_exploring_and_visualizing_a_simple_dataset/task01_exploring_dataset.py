"""Task 1: Exploring and Visualizing a Simple Dataset."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """Load the Iris dataset from seaborn."""
    return sns.load_dataset("iris")


def summarize(df: pd.DataFrame):
    """Print dataset shape, columns, and head."""
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())


def plot_pairwise(df: pd.DataFrame):
    """Create a pairplot showing relationships between Iris features."""
    sns.pairplot(df, hue="species", corner=True)
    plt.suptitle("Iris Dataset Pairwise Relationships", y=1.02)
    plt.show()


def plot_histograms(df: pd.DataFrame):
    """Plot histograms for each numeric feature."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    df["sepal_length"].hist(ax=axes[0, 0], bins=15)
    axes[0, 0].set_title("Sepal Length Distribution")
    df["sepal_width"].hist(ax=axes[0, 1], bins=15)
    axes[0, 1].set_title("Sepal Width Distribution")
    df["petal_length"].hist(ax=axes[1, 0], bins=15)
    axes[1, 0].set_title("Petal Length Distribution")
    df["petal_width"].hist(ax=axes[1, 1], bins=15)
    axes[1, 1].set_title("Petal Width Distribution")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame):
    """Plot boxplots for the numeric features."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.drop(columns="species"))
    plt.title("Boxplots for Iris Numerical Features")
    plt.show()


def main():
    df = load_data()
    summarize(df)
    plot_pairwise(df)
    plot_histograms(df)
    plot_boxplots(df)


if __name__ == "__main__":
    main()
