

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

