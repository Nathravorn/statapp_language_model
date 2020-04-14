import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_quantiles(df, levels=[0.05, 0.1, 0.25]):
    """Plot a dataframe's aggregated column values, with mean and quantiles
    around it.
    
    Args:
        df (pd.DataFrame): Dataframe to plot.
        levels (list): Quantile levels to add around the mean.
            Defaults to [0.05, 0.1, 0.25].
    
    Returns:
        Figure: matplotlib Figure object for the graph.
        Axes: matplotlib Axes object for the graph.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(df.index, df.mean(1), label="mean", lw=2)
    levels = sorted(levels, reverse=True)
    for i, level in enumerate(levels):
        plt.fill_between(
            df.index,
            df.quantile(level, axis=1),
            df.quantile(1-level, axis=1),
            color="C0",
            alpha=1/2**(i+1),
            label="{:.0%}".format(1-level),
        )
    plt.legend()
    plt.tight_layout()
    
    return fig, ax

def plot_entropy_distribution_over_layers(attentions):
    """Plot the distribution of head's entropy values over the layers of the model based on particular attention values.
    
    Args:
        attentions (np.array): Model attentions.
            Array of shape: (n_batches, n_layers, n_heads, seq_length, seq_length),
            Representing:   (batch    , layer   , head   , position  , position  ).
    
    Returns:
        Figure: matplotlib Figure object for the graph.
        Axes: matplotlib Axes object for the graph.
    """
    entropy = compute_entropy(attentions) # (batch, layer, head, position)
    average_entropy = entropy.mean(axis=-1).mean(axis=0) # (layer, head)
    df = pd.DataFrame(average_entropy)
    
    fig, ax = plot_quantiles(df)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy")
    return fig, ax
