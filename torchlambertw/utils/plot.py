"""Misc plotting functions."""


import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def test_norm(x):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first subplot (Histogram with KDE)
    sns.histplot(x, kde=True, ax=axes[0])
    axes[0].grid(True)
    axes[0].set_title("Histogram with KDE")

    # Plot the second subplot (Q-Q Plot)
    sm.qqplot(x, line="45", fit=True, ax=axes[1])
    axes[1].grid(True)
    axes[1].set_title("Q-Q Plot")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure with both subplots
    plt.show()
