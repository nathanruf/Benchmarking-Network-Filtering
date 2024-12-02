"""
This module provides benchmarking tools for evaluating network filtering algorithms.

Functions:
- __add_noise_to_network: Adds random edges to a network to simulate noise.
- bench_noise_filtering: Benchmarks a network filter's ability to reduce noise.
- bench_structural_noise_filtering: Measure the ability of net_filter to reduce noise after an initial structural filtering.
- bench_net2net_filtering: Benchmarks a network filter by applying it and calculating a net2net indicator between the original
and filtered networks.


Dependencies:
- networkx
- numpy

Usage:
This module is typically used in conjunction with custom network filtering algorithms
to evaluate their effectiveness in preserving important network structure while
removing noise.
"""

import networkx as nx
import numpy as np

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

        # Get existing weights
        existing_weights = [data['weight'] for _, _, data in noisy_net.edges(data=True) if 'weight' in data]
        
        if existing_weights:
            min_weight = min(existing_weights)
            max_weight = max(existing_weights)
        else:
            min_weight, max_weight = 0, 1  # Default range if no edges exist

        while edges_added < num_edges_to_add:
            u, v = np.random.choice(num_nodes, 2, replace=False)
            if not noisy_net.has_edge(u, v):
                if existing_weights:  # Add edge with weight if weights exist
                    noisy_net.add_edge(u, v, weight=np.random.uniform(min_weight, max_weight))
                else:  # Add edge without weight if no weights exist
                    noisy_net.add_edge(u, v)
                edges_added += 1

        return noisy_net

    def bench_noise_filtering(self, input_net: nx.Graph, net_filter: callable, indicator_funcs: list[callable],
                              noise_level: float = 0.25, seed: int = 42) -> list:
        """
        Measure ability of net_filter to reduce noise in input_net.
        Add noise to the network, apply the filter function, and calculates
        a net2net indicator from networkIndicators.py between the original and filtered networks.

        Args:
            input_net (nx.Graph): The input network.
            net_filter (callable): The network filter function.
            indicator_func (callable): The indicator function from networkIndicators.py to be used.
            noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            list: The results of the indicator functions between the filtered network and the input network.
        """
        # Add noise to the network
        noisy_net = self.__add_noise_to_network(input_net, noise_level, seed)

        # Apply the filter function
        filtered_net = net_filter(noisy_net)

        # Apply the indicator functions between the original and filtered networks
        results = []
        for indicator_func in indicator_funcs:
            results.append(indicator_func(input_net, filtered_net))
        return results

    def bench_structural_noise_filtering(self, input_net: nx.Graph, net_filter: callable, indicator_funcs: list[callable],
                                          noise_level: float = 0.25, seed: int = 42) -> list:
        """
        Measure the ability of net_filter to reduce noise after an initial structural filtering.
        First, apply the filter to the input network to obtain the structural network, then add noise to it.
        Apply the filter again to reduce the noise, and calculates a net2net indicator from networkIndicators.py 
        between the original and filtered networks.

        Args:
            input_net (nx.Graph): The input network.
            net_filter (callable): The network filter function.
            indicator_func (callable): The indicator function from networkIndicators.py to be used.
            noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            list: The results of the indicator functions between the filtered network and the input network.
        """
        # Apply the filter to get the base structural network
        structural_net = net_filter(input_net)

        # Add noise to the network
        noisy_net = self.__add_noise_to_network(structural_net, noise_level, seed)

        # Apply the filter function
        filtered_net = net_filter(noisy_net)

        # Apply the indicator functions between the original and filtered networks
        results = []
        for indicator_func in indicator_funcs:
            results.append(indicator_func(structural_net, filtered_net))
        return results