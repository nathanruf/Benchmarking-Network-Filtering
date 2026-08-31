"""
netfilterbench — benchmark network filtering and sparsification techniques.

Given a graph, apply a filter (MST, PMFG, TMFG, threshold, disparity filter,
k-core, and other sparsifiers), inject noise, and measure how well the
filter recovers the original structure using a set of comparison metrics
(Jaccard similarity, information retention, precision/recall/F1, and common
structural metrics).

Quick start:

    from netfilterbench import run_experiment, mst, bench_noise_filtering, calculate_jaccard_similarity

    result = run_experiment(
        graph_path="my_graph.pickle",
        filter_func=mst,
        benchmark_func=bench_noise_filtering,
        noise_level=0.1,
        indicator_funcs=[calculate_jaccard_similarity],
    )

See the project README for the full list of filters, benchmarks, and
indicators, and for running the pipeline across many graphs in parallel
with `run_all`.
"""

from .net_filtering import (
    to_undirected,
    mst,
    pmfg,
    tmfg,
    threshold,
    local_degree_sparsifier,
    random_edge_sparsifier,
    simmelian_sparsifier,
    disparity_filter,
    overlapping_trees,
    k_core_decomposition,
)
from .benchmark import (
    bench_noise_filtering,
    bench_structural_noise_filtering,
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
)
from .data_loader import list_graph_files
from .experiments import load_graph, run_experiment, run_all

__version__ = "0.1.0"

__all__ = [
    # filters
    "to_undirected",
    "mst",
    "pmfg",
    "tmfg",
    "threshold",
    "local_degree_sparsifier",
    "random_edge_sparsifier",
    "simmelian_sparsifier",
    "disparity_filter",
    "overlapping_trees",
    "k_core_decomposition",
    # benchmarks
    "bench_noise_filtering",
    "bench_structural_noise_filtering",
    # indicators
    "calculate_information_retention",
    "calculate_jaccard_similarity",
    "common_metrics",
    "predictive_filtering_metrics",
    # data loading / experiment orchestration
    "list_graph_files",
    "load_graph",
    "run_experiment",
    "run_all",
]
