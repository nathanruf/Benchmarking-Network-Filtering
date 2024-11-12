"""
This module provides functions for calculating various network indicators
using NetworkX graphs. These indicators can be used to analyze and compare
different network structures.
Functions:
- calculate_information_retention: Calculates the Information Retention score
  between an original and a filtered network.
- calculate_network_similarity: Calculates the similarity between two networks
  based on common network metrics.
- calculate_jaccard_distance: Calculate the Jaccard distance between two networks based on their edge sets.
"""

import networkx as nx
import numpy as np
import warnings
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, root_mean_squared_error
from typing import Union, List
from scipy.stats import entropy
from itertools import combinations

class NetworkIndicators():
    def calculate_information_retention(self, original_graph: Union[nx.Graph, nx.DiGraph], 
								filtered_graph: Union[nx.Graph, nx.DiGraph]) -> float:
        """
        Calculate the Information Retention score of a filtered network.

        This function compares the degree distributions of the original and filtered
        networks using the Kullback-Leibler divergence (relative entropy).

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network
            filtered_graph (Union[nx.Graph, nx.DiGraph]): The filtered network

        Returns:
        float: Information Retention score (higher is better)
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
    
    def predictive_filtering_metrics(self, original_graph: Union[nx.Graph, nx.DiGraph],
                                 filtered_graph: Union[nx.Graph, nx.DiGraph]) -> dict:
        """
        Calculate predictive filtering metrics for the given graphs.

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network.
            filtered_graph (Union[nx.Graph, nx.DiGraph]): The filtered network.

        Returns:
            dict: A dictionary containing calculated metrics, such as true positives, false negatives, precision, recall, F1 score and RMSE.
        """
        y_true = []  # True labels for the edges
        y_pred = []  # Predicted labels for the edges

        # Adds 1 if edge (u, v) exists, else 0
        for u, v in combinations(original_graph.nodes, 2):
            y_true.append(1 if original_graph.has_edge(u, v) else 0)
            y_pred.append(1 if filtered_graph.has_edge(u, v) else 0)

        # Names of the metrics
        metric_names = [
            'true_positive',
            'true_negative',
            'false_positive',
            'false_negative',
            'precision',
            'recall',
            'f1_score',
            'RMSE'
        ]

        # Calculate the confusion matrix
        confusion_matrix_values = confusion_matrix(y_true, y_pred, labels = [0, 1])
        
        # Extract the metrics from the confusion matrix and calculate the other metrics
        metrics_values = [
            confusion_matrix_values[1][1],  # True positives
            confusion_matrix_values[0][0],  # True negatives
            confusion_matrix_values[0][1],  # False positives
            confusion_matrix_values[1][0],  # False negatives
            precision_score(y_true, y_pred, zero_division = 0),   # Precision
            recall_score(y_true, y_pred, zero_division = 0),      # Recall
            f1_score(y_true, y_pred, zero_division = 0),          # F1 Score
            root_mean_squared_error(y_true, y_pred)               # RMSE
        ]

        res = {}  # Dictionary to store the results

        # Map metrics values to their corresponding names
        for value, name in zip(metrics_values, metric_names):
            res[name] = value

        return res  # Return the dictionary containing metrics

    
    def common_metrics(self, original_graph: Union[nx.Graph, nx.DiGraph], 
                         filtered_graph: Union[nx.Graph, nx.DiGraph]) -> dict:
        """
        Calculate common network metrics for the original and filtered graphs.

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network.
            filtered_graph (Union[nx.Graph, nx.DiGraph]): The filtered network.

        Returns:
            dict: A dictionary with the calculated metrics.
        """

        def calculate_metrics(G: Union[nx.Graph, nx.DiGraph]) -> List[float]:
            """
            Calculate common network metrics for a given graph.

            Parameters:
            G (Union[nx.Graph, nx.DiGraph]): The input network

            Returns:
            List[float]: A list of calculated metrics
            """
            metrics = []

            # List of metric functions
            metric_funcs = [
                nx.degree,
                nx.average_clustering,
                nx.average_shortest_path_length,
                nx.diameter,
                nx.betweenness_centrality,
                nx.closeness_centrality,
                nx.global_efficiency,
                nx.degree_assortativity_coefficient,
                nx.density,
                nx.transitivity
            ]

            for metric_func in metric_funcs:
                try:
                    # Check if the graph has no edges
                    if G.number_of_edges() == 0:
                        metrics.append(None)  # Return None if the graph has no edges
                        continue

                    result = metric_func(G)
                    # If the result is a dict, take its mean
                    if isinstance(result, dict):
                        result = np.mean(list(result.values())) if result else None
                    metrics.append(result)
                except Exception:
                    metrics.append(None)  # Append None if an exception occurs

            degrees = [d for n, d in G.degree()]
            metrics.append(np.var(degrees)) # Degree Variance
            metrics.append(max(degrees)) # Maximum Degree

            return metrics
        
        # Names of the metrics
        filtered_metric_names = [
            'average_degree_filtered',
            'average_clustering_filtered',
            'average_path_length_filtered',
            'diameter_filtered',
            'average_betweenness_filtered',
            'average_closeness_filtered',
            'global_efficiency_filtered',
            'degree_assortativity_filtered',
            'density_filtered',
            'transitivity_filtered',
            'degree_variance_filtered',
            'maximum_degree_filtered'
        ]

        original_metric_names = [
            'average_degree_original',
            'average_clustering_original',
            'average_path_length_original',
            'diameter_original',
            'average_betweenness_original',
            'average_closeness_original',
            'global_efficiency_original',
            'degree_assortativity_original',
            'density_original',
            'transitivity_original',
            'degree_variance_original',
            'maximum_degree_original'
        ]
        
        # Dictionary to store the ratios
        results_dict = {}

        # Calculate metrics for both networks
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            original_metrics = calculate_metrics(original_graph)
            filtered_metrics = calculate_metrics(filtered_graph)

        # Populate results_dict with calculated metrics
        for original_name, filtered_name, original_metric, filtered_metric in zip(original_metric_names, filtered_metric_names,
                                                                                   original_metrics, filtered_metrics):
            results_dict[original_name] = original_metric
            results_dict[filtered_name] = filtered_metric

        return results_dict
        
    def calculate_jaccard_similarity(self, network1: Union[nx.Graph, nx.DiGraph], network2: Union[nx.Graph, nx.DiGraph]) -> float:
        """
        Calculate the Jaccard similarity between two networks based on their edge sets.

        The Jaccard similarity is defined as 1 minus the Jaccard similarity coefficient.
        It measures the similarity between two sets, in this case, the edge sets of the networks.

        Parameters:
            network1 (Union[nx.Graph, nx.DiGraph]): The first network
            network2 (Union[nx.Graph, nx.DiGraph]): The second network

        Returns:
        float: Jaccard similarity between the two networks (lower is more similar)
        """
        try:
            # Get edge sets for both networks
            edges1 = set(network1.edges())
            edges2 = set(network2.edges())

            # Calculate Jaccard similarity coefficient
            intersection = len(edges1.intersection(edges2))
            union = len(edges1.union(edges2))

            jaccard_similarity = intersection / union if union > 0 else 0

            return jaccard_similarity

        except Exception as e:
            print(f"Error calculating Jaccard distance: {str(e)}")
            return None