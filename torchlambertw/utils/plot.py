"""Misc plotting functions."""


import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy
import scipy.stats
import numpy as np


def test_norm(x):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first subplot (Histogram with KDE)
    sns.histplot(
        x,
        kde=True,
        ax=axes[0],
        stat="density",
        edgecolor="black",
        color="lightgray",
    )
    sns.kdeplot(data=x, color="black", lw=2, ax=axes[0])
    axes[0].grid(True)
    axes[0].set_title("Density Estimates")

    # Calculate the theoretical normal distribution PDF
    mu, std = np.mean(x), np.std(x)
    xmin, xmax = axes[0].get_xlim()
    x_range = np.linspace(xmin, xmax, 100)
    pdf = scipy.stats.norm.pdf(x_range, loc=mu, scale=std)  # Use scipy.stats.norm.pdf
    # Plot the theoretical normal distribution as a red dashed line and add it to the legend
    (line1,) = axes[0].plot(
        x_range,
        pdf,
        label=f"N({mu:.2f}, {std:.2f}^2)",
        color="red",
        linestyle="--",
    )

    # Add skewness and kurtosis estimates to the upper-left area of the chart
    skewness = scipy.stats.skew(x)
    kurtosis = scipy.stats.kurtosis(x) + 3
    axes[0].text(
        0.05,
        0.95,
        f"Skewness: {skewness:.1f}\nKurtosis: {kurtosis:.1f}",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    # Plot the second subplot (Q-Q Plot)
    sm.qqplot(x, line="45", fit=True, ax=axes[1], color="black", fmt="k--")
    axes[1].grid(True)
    axes[1].set_title("Q-Q Plot (Normal)")

    # Add a legend to the upper right corner of the first subplot
    axes[0].legend(handles=[line1], loc="upper right")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure with both subplots
    plt.show()
