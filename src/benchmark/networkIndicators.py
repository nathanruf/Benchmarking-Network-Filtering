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
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from typing import Union, List
from scipy.stats import entropy
from scipy.spatial.distance import cosine

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
                                 noisy_graph: Union[nx.Graph, nx.DiGraph],
                                 filtered_graph: Union[nx.Graph, nx.DiGraph]) -> dict:
        """
        Calculate predictive filtering metrics for the given graphs.

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network.
            noisy_graph (Union[nx.Graph, nx.DiGraph]): The graph with added noise.
            filtered_graph (Union[nx.Graph, nx.DiGraph]): The filtered network.

        Returns:
            dict: A dictionary containing calculated metrics, such as true positives, false negatives, precision, recall, and F1 score.
        """
        y_true = []  # True labels for the edges
        y_pred = []  # Predicted labels for the edges

        original_edges = original_graph.edges()  # Edges of the original graph
        filtered_edges = filtered_graph.edges()  # Edges of the filtered graph
        noisy_edges = noisy_graph.edges()        # Edges of the noisy graph

        # Populate y_true and y_pred based on the edges
        for edge in noisy_edges:
            y_true.append(1 if edge in original_edges else 0)  # 1 if edge exists in original, else 0
            y_pred.append(1 if edge in filtered_edges else 0)  # 1 if edge exists in filtered, else 0

        # Names of the metrics
        metric_names = [
            'true_positive',
            'true_negative',
            'false_positive',
            'false_negative',
            'precision',
            'recall',
            'f1_score'
        ]

        # Calculate the confusion matrix
        confusion_matrix_values = confusion_matrix(y_true, y_pred, labels = [0, 1])
        
        # Extract the metrics from the confusion matrix and calculate the other metrics
        metrics_values = [
            confusion_matrix_values[1][1],  # True positives
            confusion_matrix_values[0][0],  # True negatives
            confusion_matrix_values[0][1],  # False positives
            confusion_matrix_values[1][0],  # False negatives
            precision_score(y_true, y_pred, zero_division = 0),  # Precision
            recall_score(y_true, y_pred, zero_division = 0),      # Recall
            f1_score(y_true, y_pred, zero_division = 0)           # F1 Score
        ]

        res = {}  # Dictionary to store the results

        # Map metrics values to their corresponding names
        for value, name in zip(metrics_values, metric_names):
            res[name] = value

        return res  # Return the dictionary containing metrics

    
    def common_metrics_ratio(self, original_graph: Union[nx.Graph, nx.DiGraph], 
                         filtered_graph: Union[nx.Graph, nx.DiGraph]) -> dict:
        """
        Calculate the ratio of common network metrics between the original and filtered graphs,
        adding 1 to each metric to prevent division by zero.

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network.
            filtered_graph (Union[nx.Graph, nx.DiGraph]): The filtered network.

        Returns:
            List[float]: A list of ratios for calculated metrics, comparing the filtered graph to the original graph.
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
                nx.degree_assortativity_coefficient,
                nx.average_clustering,
                nx.average_shortest_path_length,
                nx.density,
                nx.average_degree_connectivity,
                nx.transitivity
            ]

            for metric_func in metric_funcs:
                try:
                    # Check if the graph has no edges
                    if G.number_of_edges() == 0:
                        metrics.append(None)  # Return None if the graph has no edges
                        continue

                    result = metric_func(G)
                    # If the result is a dict (only for average_degree_connectivity), take its mean
                    if isinstance(result, dict):
                        result = sum(result.values()) / len(result) if result else None
                    metrics.append(result)
                except Exception:
                    metrics.append(None)  # Append None if an exception occurs

            return metrics
        
        # Names of the metrics
        metric_names = [
            'degree_assortativity',
            'average_clustering',
            'average_shortest_path_length',
            'density',
            'average_degree_connectivity',
            'transitivity'
        ]
        
        # Dictionary to store the ratios
        ratio_dict = {}

        # Calculate metrics for both networks
        original_metrics = calculate_metrics(original_graph)
        filtered_metrics = calculate_metrics(filtered_graph)

        # Calculate ratios
        for name, original, filtered in zip(metric_names, original_metrics, filtered_metrics):
            if original is not None and filtered is not None:  # Check for valid metric values
                ratio_dict[name] = (filtered + 1) / (original + 1)
            else:
                ratio_dict[name] = None  # Calculate the ratio with normalization

        return ratio_dict
        
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