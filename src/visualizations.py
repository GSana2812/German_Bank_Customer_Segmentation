import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List

def scatters(data, cols = None, h=None, pal=None):
    """
    Generate scatter plots for visualizing relationships between variables.

    Parameters:
        data (DataFrame): The dataset to be visualized.
        h (str): Column for hue (coloring points based on a categorical variable).
        pal (str): Color palette for the plot.
    """
    # Create a 3x1 subplot layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    # Scatter plot for Credit amount vs. Duration
    sns.scatterplot(x=cols[0], y=cols[1], hue=h, palette=pal, data=data, ax=ax1)

    # Scatter plot for Age vs. Credit amount
    sns.scatterplot(x=cols[2], y=cols[0], hue=h, palette=pal, data=data, ax=ax2)

    # Scatter plot for Age vs. Duration
    sns.scatterplot(x=cols[2], y=cols[1], hue=h, palette=pal, data=data, ax=ax3)

    # Adjust layout for better spacing
    plt.tight_layout()

def joint_plot_pearson(data, x = None, y = None):

    r1 = sns.jointplot(x=x, y=y, data=data, kind="reg", height=8)

    # Calculate Pearson correlation coefficient
    pearson_corr, _ = stats.pearsonr(data[x], data[y])

    # Annotate the plot with the Pearson correlation coefficient
    r1.ax_joint.annotate(f"Pearson Corr: {pearson_corr:.2f}", xy=(0.6, 0.9), xycoords="axes fraction")

    # Display the plot
    plt.show()

def count_purposes(data, x = None, group_col=None):
    # Group the data by "Purpose" and count the number of credits for each purpose
    n_credits = data.groupby(x)[group_col].count().rename("Count").reset_index()

    # Sort purposes by the count of granted credits in descending order
    n_credits.sort_values(by=["Count"], ascending=False, inplace=True)

    # Create a bar plot to visualize the number of granted credits by purpose
    plt.figure(figsize=(10, 6))
    bar = sns.barplot(x=x, y="Count", data=n_credits)

    # Rotate x-axis labels for better readability
    bar.set_xticklabels(bar.get_xticklabels(), rotation=60)

    # Set y-axis label
    plt.ylabel("Number of granted credits")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

def boxes(data, x, y, h, rotation=45):
    """
    Generate a box plot for visualizing distribution of variables.

    Parameters:
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis.
        h (str): Column name for hue (coloring boxes based on a categorical variable).
        rotation (int): Rotation angle for x-axis labels.
    """

    # Create a figure and axis for the box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the box plot with specified parameters
    box = sns.boxplot(x=x, y=y, hue=h, data=data)

    # Rotate x-axis labels for better readability
    box.set_xticklabels(box.get_xticklabels(), rotation=rotation)

    # Adjust layout for better spacing
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()

def distributions(df, cols = None):

    """
    Generate distribution plots for selected variables.

    Parameters:
        df (DataFrame): The dataset containing the variables to be visualized.
    """


    # Create a 3x1 subplot layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    # Plot distribution of "Credit amount"
    sns.distplot(df[cols[0]], ax=ax2)

    # Plot distribution of "Duration"
    sns.distplot(df[cols[1]], ax=ax3)

    # Plot distribution of "Age"
    sns.distplot(df[cols[2]], ax=ax1)

    # Adjust layout for better spacing
    plt.tight_layout()

def plot_inertias(clusters_range:List[int], inertias: List[int])-> None:


    plt.plot(clusters_range, inertias, marker='o')
    plt.show()

def plot_pivot(pivot_km: pd.pivot_table)-> None:

    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm.rocket_r)
    plt.tight_layout()