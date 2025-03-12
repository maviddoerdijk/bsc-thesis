import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_time_series(df: pd.DataFrame, columns: list, title: str = "Time Series Plot"):
    plt.figure(figsize=(12,6))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss(history, title: str = "Training vs. Validation Loss"):
    plt.figure(figsize=(10,5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_heatmap(data, title: str = "Heatmap"):
    plt.figure(figsize=(10,8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

def plot_bar_chart(categories, values, title="Bar Chart", xlabel="Model", ylabel="Metric"):
    plt.figure(figsize=(8,6))
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    # Simple test plots
    import numpy as np
    import pandas as pd
    dates = pd.date_range("2021-01-01", periods=50)
    df = pd.DataFrame({"A": np.sin(np.linspace(0, 10, 50)), "B": np.cos(np.linspace(0, 10, 50))}, index=dates)
    plot_time_series(df, ["A", "B"], title="Sine and Cosine")