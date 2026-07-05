"""
This module provides functions for calculating various network indicators
using NetworkX graphs. These indicators can be used to analyze and compare
different network structures.

Functions:
- calculate_information_retention: Calculates a KL-based similarity score
  between the degree distributions of the original and filtered networks.
- predictive_filtering_metrics: Computes edge prediction metrics (TP, TN, FP, FN, precision, recall, F1, RMSE).
- common_metrics: Computes and compares common structural network metrics.
- calculate_jaccard_similarity: Calculates the Jaccard similarity between two networks based on their edge sets.

Note:
    All functions expect undirected graphs (nx.Graph). Directed graphs (nx.DiGraph)
    should be converted before being passed to these functions.
"""

import networkx as nx
import numpy as np
import warnings
from typing import List, Union
from scipy.stats import entropy


def calculate_information_retention(original_graph: nx.Graph,
                                    filtered_graph: nx.Graph) -> float:
    """
    Calculate the Information Retention score of a filtered network.

    This function compares the degree distributions of the original and filtered
    networks using the Kullback-Leibler divergence (relative entropy).

    Parameters:
        original_graph (nx.Graph): The original, unfiltered network.
        filtered_graph (nx.Graph): The filtered network.

    Returns:
        float: Information Retention score (higher is better).
    """
    # Calculate degree distributions
    original_degrees = [d for _, d in original_graph.degree()]
    filtered_degrees = [d for _, d in filtered_graph.degree()]

    # Get the range of degrees
    max_degree = max(max(original_degrees), max(filtered_degrees))
    degree_range = range(max_degree + 1)

    # Calculate degree histograms
    original_hist, _ = np.histogram(original_degrees, bins=degree_range, density=True)
    filtered_hist, _ = np.histogram(filtered_degrees, bins=degree_range, density=True)

    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    original_hist += epsilon
    filtered_hist += epsilon

    # Normalize histograms
    original_hist /= original_hist.sum()
    filtered_hist /= filtered_hist.sum()

    # Calculate Kullback-Leibler divergence
    kl_divergence = entropy(original_hist, filtered_hist)

    # Convert divergence to a similarity score
    information_retention = np.exp(-kl_divergence)

    return information_retention


def predictive_filtering_metrics(original_graph: nx.Graph,
                                 filtered_graph: nx.Graph) -> dict:
    """
    Calculate predictive filtering metrics for the given graphs.

    Uses set operations on edge sets for efficiency, avoiding O(n^2) iteration
    over all node pairs.

    Parameters:
        original_graph (nx.Graph): The original, unfiltered network.
        filtered_graph (nx.Graph): The filtered network.

    Returns:
        dict: A dictionary containing calculated metrics: true positives, true negatives,
              false positives, false negatives, precision, recall, F1 score, and RMSE.
    """
    edges_orig = set(original_graph.edges())
    edges_filt = set(filtered_graph.edges())

    n = original_graph.number_of_nodes()
    total_pairs = n * (n - 1) // 2

    tp = len(edges_orig & edges_filt)
    fp = len(edges_filt - edges_orig)
    fn = len(edges_orig - edges_filt)
    tn = total_pairs - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rmse      = np.sqrt((fp + fn) / total_pairs) if total_pairs > 0 else 0.0

    return {
        'true_positive':  tp,
        'true_negative':  tn,
        'false_positive': fp,
        'false_negative': fn,
        'precision':      precision,
        'recall':         recall,
        'f1_score':       f1,
        'RMSE':           rmse,
    }


def common_metrics(original_graph: nx.Graph,
                   filtered_graph: nx.Graph,
                   betweenness_k: int = None) -> dict:
    """
    Calculate common network metrics for the original and filtered graphs.

    Parameters:
        original_graph (nx.Graph): The original, unfiltered network.
        filtered_graph (nx.Graph): The filtered network.

    Returns:
        dict: A dictionary with the calculated metrics.
    """

    def calculate_metrics(G: nx.Graph) -> List[float]:
        """
        Calculate common network metrics for a given graph.

        Parameters:
            G (nx.Graph): The input network.

        Returns:
            List[float]: A list of calculated metrics.
        """
        metrics = []

        metric_funcs = [
            nx.degree,
            nx.average_clustering,
            nx.average_shortest_path_length,
            nx.diameter,
            nx.closeness_centrality,
            nx.global_efficiency,
            nx.degree_assortativity_coefficient,
            nx.density,
            nx.transitivity
        ]

        for metric_func in metric_funcs:
            try:
                if G.number_of_edges() == 0:
                    metrics.append(None)
                    continue
                result = metric_func(G)
                if isinstance(result, dict):
                    result = np.mean(list(result.values())) if result else None
                elif metric_func == nx.degree:
                    result = np.mean([degree for _, degree in result])
                metrics.append(result)
            except Exception:
                metrics.append(None)

        # Betweenness centrality is computed separately to support an optional k-sample
        # approximation. Pass betweenness_k=<int> for approximate (faster) computation,
        # or omit for exact computation (default).
        try:
            if G.number_of_edges() == 0:
                metrics.append(None)
            else:
                bc = nx.betweenness_centrality(G, normalized=True, k=betweenness_k)
                metrics.append(np.mean(list(bc.values())))
        except Exception:
            metrics.append(None)

        degrees = [d for _, d in G.degree()]
        metrics.append(np.var(degrees))
        metrics.append(max(degrees))

        return metrics

    filtered_metric_names = [
        'average_degree_filtered',
        'average_clustering_filtered',
        'average_path_length_filtered',
        'diameter_filtered',
        'average_closeness_filtered',
        'global_efficiency_filtered',
        'degree_assortativity_filtered',
        'density_filtered',
        'transitivity_filtered',
        'average_betweenness_filtered',
        'degree_variance_filtered',
        'maximum_degree_filtered'
    ]

    original_metric_names = [
        'average_degree_original',
        'average_clustering_original',
        'average_path_length_original',
        'diameter_original',
        'average_closeness_original',
        'global_efficiency_original',
        'degree_assortativity_original',
        'density_original',
        'transitivity_original',
        'average_betweenness_original',
        'degree_variance_original',
        'maximum_degree_original'
    ]

    results_dict = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        original_metrics = calculate_metrics(original_graph)
        filtered_metrics = calculate_metrics(filtered_graph)

    for original_name, filtered_name, original_metric, filtered_metric in zip(
        original_metric_names, filtered_metric_names, original_metrics, filtered_metrics
    ):
        results_dict[original_name] = original_metric
        results_dict[filtered_name] = filtered_metric

    return results_dict


def calculate_jaccard_similarity(network1: nx.Graph, network2: nx.Graph) -> float:
    """
    Calculate the Jaccard similarity between two networks based on their edge sets.

    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the edge sets.

    Parameters:
        network1 (nx.Graph): The first network.
        network2 (nx.Graph): The second network.

    Returns:
        float: Jaccard similarity between the two networks (higher is more similar).
    """
    try:
        edges1 = set(network1.edges())
        edges2 = set(network2.edges())

        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)

        return intersection / union if union > 0 else 0.0

    except Exception as e:
        print(f"Error calculating Jaccard similarity: {str(e)}")
        return None