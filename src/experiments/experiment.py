import pickle
import traceback
import os
from typing import Callable, Dict, Any, List
from pathlib import Path

from src.benchmark.networkIndicators import (
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
)


def load_graph(path: str):
    """
    Load a graph from file based on extension.
    Currently supports pickle. Extendable.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pickle":
        with open(path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported file format: {ext}")


def _flatten_results(results: dict) -> dict:
    """
    Flatten the results dict returned by benchmark functions.

    Functions that return a nested dict (e.g. common_metrics, predictive_filtering_metrics)
    are merged into the top-level row. Functions that return a scalar are kept as-is,
    using the function name as key.
    """
    row = {}
    for key, value in results.items():
        if isinstance(value, dict):
            row.update(value)
        else:
            row[key] = value
    return row


def run_experiment(
    graph_path: str,
    filter_func: Callable,
    benchmark_func: Callable,
    noise_level: float,
    indicator_funcs: List[Callable],
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        graph_path (str): Path to the graph file.
        filter_func (Callable): The network filter function.
        benchmark_func (Callable): The benchmark function (e.g. bench_noise_filtering).
        noise_level (float): Noise level to apply.
        indicator_funcs (list[Callable]): Indicator functions to compute.

    Returns:
        dict: All computed metrics. On failure, dumps a debug pickle and re-raises.
    """
    base_row = {
        "graph_name": Path(graph_path).stem,
        "filter": filter_func.__name__,
        "benchmark": benchmark_func.__name__,
        "noise_level": noise_level,
    }

    try:
        graph = load_graph(graph_path)

        results = benchmark_func(
            graph,
            filter_func,
            indicator_funcs,
            noise_level=noise_level,
        )

        return {**base_row, **_flatten_results(results)}

    except Exception as e:
        with open("debug_failed_experiment.pkl", "wb") as f:
            pickle.dump(
                {
                    "graph_path": graph_path,
                    "filter": filter_func.__name__,
                    "benchmark": benchmark_func.__name__,
                    "noise_level": noise_level,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                f,
            )
        raise