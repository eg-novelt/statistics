

from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import reveal_type
import arviz as az
import xarray
import graphviz as gr

import asyncpg 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sample_or_load(
    sample_func: Callable[[], xarray.DataTree],
    cache_file: Path,
    clean : bool = False 
) -> xarray.DataTree:
    """Load cached inference results or run sampling and cache them.

    Args:
        sample_func: Zero-argument function that runs sampling and
            returns an ArviZ InferenceData object.
        cache_file: Path to the NetCDF file used to cache the results.

    Returns:
        The loaded or newly sampled InferenceData.
    """
    cache_file = Path(cache_file)

    # Reuse existing results to avoid rerunning expensive sampling.
    if cache_file.exists() and not clean:
        idata= az.from_netcdf(cache_file)
        #print(reveal_type(idata))
        return idata

    # Run sampling only when no cached results exist.
    idata = sample_func()

    # Persist the results so they can be reused in later runs.
    idata.to_netcdf(cache_file)

    #print(reveal_type(idata))

    

    return idata

def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph"""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g

def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)

def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    del background_plot_kwargs["label"]  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)

