"""
This module provides benchmarking tools for evaluating network filtering algorithms.

Functions:
- add_noise_to_network: Adds random edges to a network to simulate noise.
- bench_noise_filtering: Benchmarks a network filter's ability to reduce noise.
- bench_structural_noise_filtering: Measure the ability of net_filter to reduce noise after an initial structural filtering.
- calculate_information_retention: Calculates the Information Retention score between an original and a filtered network.
- calculate_network_similarity: Calculates the similarity between an original and a filtered network based on common network metrics.
- calculate_jaccard_distance: Calculate the Jaccard distance between an original and a filtered network based on their edge sets.


Dependencies:
- networkx
- numpy
- scikit-learn
- scipy
- typing

Usage:
This module is typically used in conjunction with custom network filtering algorithms
to evaluate their effectiveness in preserving important network structure while
removing noise.
"""

import networkx as nx
import numpy as np
from sklearn.metrics import jaccard_score
from typing import Union, List
from scipy.stats import entropy
from scipy.spatial.distance import cosine

class Benchmark:
    def __add_noise_to_network(self, input_net: nx.Graph, noise_level: float, seed: int = 42) -> nx.Graph:
        """
        Add noise to the network by adding random edges between unconnected vertices.

        Args:
            input_net (nx.Graph): The input network.
            noise_level (float): The ratio of random edges to total edges in the resulting graph.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            nx.Graph: The network with added noise.
        """
        np.random.seed(seed)
        noisy_net = input_net.copy()
        num_nodes = noisy_net.number_of_nodes()
        current_edges = noisy_net.number_of_edges()
        
        # Calculate the number of edges to add
        total_edges_after_noise = int(current_edges / (1 - noise_level))
        num_edges_to_add = total_edges_after_noise - current_edges
        
        # Ensure we don't exceed the maximum possible edges
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_edges_to_add = min(num_edges_to_add, max_possible_edges - current_edges)

        edges_added = 0
        while edges_added < num_edges_to_add:
            u, v = np.random.choice(num_nodes, 2, replace=False)
            if not noisy_net.has_edge(u, v):
                noisy_net.add_edge(u, v, weight=np.random.uniform(0, 1))
                edges_added += 1

        return noisy_net

    def bench_noise_filtering(self, input_net: nx.Graph, net_filter: callable, noise_level: float = 0.25, seed: int = 42) -> float:
        """
        Measure ability of net_filter to reduce noise in input_net.
        Add noise to the network, apply the filter function, and calculate the Jaccard score.

        Args:
            input_net (nx.Graph): The input network.
            net_filter (callable): The network filter function.
            noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            float: The Jaccard score between the edge sets of the filtered network and the input network.
        """
        # Add noise to the network
        noisy_net = self.__add_noise_to_network(input_net, noise_level, seed)

        # Apply the filter function
        filtered_net = net_filter(noisy_net)

        # Calculate the Jaccard score between the edge sets
        input_edges = set(input_net.edges)
        filtered_edges = set(filtered_net.edges)
        jaccard_score = len(input_edges & filtered_edges) / len(input_edges | filtered_edges)

        return jaccard_score

    def bench_structural_noise_filtering(self, input_net: nx.Graph, net_filter: callable, noise_level: float = 0.25, seed: int = 42) -> float:
        """
        Measure the ability of net_filter to reduce noise after an initial structural filtering.
        First, apply the filter to the input network to obtain the structural network, then add noise to it.
        Apply the filter again to reduce the noise, and calculate the Jaccard score between the filtered network and the structural network.

        Args:
            input_net (nx.Graph): The input network.
            net_filter (callable): The network filter function.
            noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            float: The Jaccard score between the edge sets of the filtered network and the input network.
        """
        # Apply the filter to get the base structural network
        structural_net = net_filter(input_net)

        # Add noise to the network
        noisy_net = self.__add_noise_to_network(structural_net, noise_level, seed)

        # Apply the filter function
        filtered_net = net_filter(noisy_net)

        # Calculate the Jaccard score between the edge sets
        structural_edges = set(structural_net.edges)
        filtered_edges = set(filtered_net.edges)

        # Check for division by zero
        if len(structural_edges) == 0 and len(filtered_edges) == 0:
            return None
        
        jaccard_score = len(structural_edges & filtered_edges) / len(structural_edges | filtered_edges)

        return jaccard_score

    def calculate_information_retention(self, original_graph: Union[nx.Graph, nx.DiGraph], net_filter: callable) -> float:
        """
        Calculate the Information Retention score of a filtered network.

        This function compares the degree distributions of the original and filtered
        networks using the Kullback-Leibler divergence (relative entropy).

        Parameters:
        original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network
        net_filter (callable): The network filter function.

        Returns:
        float: Information Retention score (higher is better)
        """
        # Apply the filter function
        filtered_graph = net_filter(original_graph)

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
    
    def calculate_network_similarity(self, original_graph: Union[nx.Graph, nx.DiGraph], net_filter: callable) -> float:
        """
        Calculate the similarity between two networks based on common network metrics.

        This function computes various network metrics for both input networks,
        creates vectors of these metrics, and then calculates the cosine similarity
        between these vectors.

        Parameters:
        original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network
        net_filter (callable): The network filter function.

        Returns:
        float: Similarity score between the two networks (higher is more similar)
        """

        def calculate_metrics(G: Union[nx.Graph, nx.DiGraph]) -> List[float]:
            """
            Calculate common network metrics for a given graph.

            Parameters:
            G (Union[nx.Graph, nx.DiGraph]): The input network

            Returns:
            List[float]: A list of calculated metrics
            """
            metrics = [
                nx.degree_assortativity_coefficient(G),
                nx.average_clustering(G),
                nx.average_shortest_path_length(G),
                nx.density(G),
                nx.average_degree_connectivity(G),
                nx.transitivity(G)
            ]
            return metrics

        try:
            # Apply the filter function
            filtered_graph = net_filter(original_graph)

            # Calculate metrics for both networks
            original_metrics = calculate_metrics(original_graph)
            filtered_metrics = calculate_metrics(filtered_graph)

            # Calculate cosine similarity
            similarity = 1 - cosine(original_metrics, filtered_metrics)  # Convert distance to similarity

            return similarity

        except Exception as e:
            print(f"Error calculating network similarity: {str(e)}")
            return None
	

    def calculate_jaccard_distance(self, original_graph: Union[nx.Graph, nx.DiGraph], net_filter: callable) -> float:
        """
        Calculate the Jaccard distance between two networks based on their edge sets.

        The Jaccard distance is defined as 1 minus the Jaccard similarity coefficient.
        It measures the dissimilarity between two sets, in this case, the edge sets of the networks.

        Parameters:
            original_graph (Union[nx.Graph, nx.DiGraph]): The original, unfiltered network
            net_filter (callable): The network filter function.

        Returns:
        float: Jaccard distance between the two networks (lower is more similar)
        """
        try:
            # Apply the filter function
            filtered_graph = net_filter(original_graph)

            # Get edge sets for both networks
            original_edges = set(original_graph.edges())
            filtered_edges = set(filtered_graph.edges())

            # Calculate Jaccard similarity coefficient
            intersection = len(original_edges.intersection(filtered_edges))
            union = len(original_edges.union(filtered_edges))

            jaccard_similarity = intersection / union if union > 0 else 0

            # Calculate Jaccard distance
            jaccard_distance = 1 - jaccard_similarity

            return jaccard_distance

        except Exception as e:
            print(f"Error calculating Jaccard distance: {str(e)}")
            return None