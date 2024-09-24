"""
This module provides benchmarking tools for evaluating network filtering algorithms.

Functions:
- add_noise_to_network: Adds random edges to a network to simulate noise.
- bench_noise_filtering: Benchmarks a network filter's ability to reduce noise.

Dependencies:
- networkx
- numpy
- scikit-learn

Usage:
This module is typically used in conjunction with custom network filtering algorithms
to evaluate their effectiveness in preserving important network structure while
removing noise.
"""

import networkx as nx
import numpy as np
from sklearn.metrics import jaccard_score

def add_noise_to_network(input_net: nx.Graph, noise_level: float, seed: int = 42) -> nx.Graph:
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
			noisy_net.add_edge(u, v)
			edges_added += 1

	return noisy_net

def bench_noise_filtering(input_net: nx.Graph, net_filter: callable, noise_level: float = 0.25, seed: int = 42) -> float:
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
	noisy_net = add_noise_to_network(input_net, noise_level, seed)

	# Apply the filter function
	filtered_net = net_filter(noisy_net)

	# Calculate the Jaccard score between the edge sets
	input_edges = set(input_net.edges)
	filtered_edges = set(filtered_net.edges)
	jaccard_score = len(input_edges & filtered_edges) / len(input_edges | filtered_edges)

	return jaccard_score

def bench_net2net_filtering(input_net: nx.Graph, net_filter: callable, indicator_func: callable) -> float:
	"""
	Apply a network filter to the input network and calculate a network2network indicator.

	This function applies the given network filter to the input network,
	then calculates a net2net indicator from net2net_indicators.py between the original and filtered networks.

	Args:
		input_net (nx.Graph): The original input network.
		net_filter (callable): The network filter function to be applied.
		indicator_func (callable): The indicator function from net2net_indicators.py to be used.

	Returns:
		float: The result of the indicator function applied to the original and filtered networks.
	"""
	# Apply the filter function to generate the filtered network
	filtered_net = net_filter(input_net)

	# Apply the indicator function between the original and filtered networks
	indicator_result = indicator_func(input_net, filtered_net)

	return indicator_result




