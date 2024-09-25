"""
This module implements network filtering and graph sparsification techniques.

Functions:
- global_threshold_filter: Filters edges based on a global threshold
- local_degree_sparsifier: Sparsifies graph based on local node degrees
- random_edge_sparsifier: Randomly sparsifies edges
- simmelian_sparsifier: Implements Simmelian backbone sparsification
- disparity_filter: Implements the disparity filter technique
- overlapping_trees: Implements the Overlapping Trees network reduction technique

Dependencies:
- networkx
- numpy
- scipy
"""

import networkx as nx
import scipy
from typing import Union, List
import numpy as np

def global_threshold_filter(G: nx.Graph, 
                            attribute: str,
                            threshold: float, 
                            above: bool = True) -> nx.Graph:
	"""
	Filters edges globally using a constant threshold value and a given edge attribute.

	Args:
		G (nx.Graph): Input graph
		attribute (str): Edge attribute to use for filtering
		threshold (float): Threshold value
		above (bool): If True, keep edges with attribute value >= threshold. 
					  If False, keep edges with attribute value < threshold.

	Returns:
		nx.Graph: Filtered graph
	"""
	H = G.copy()
	for u, v, data in G.edges(data=True):
		if above and data.get(attribute, 0) < threshold:
			H.remove_edge(u, v)
		elif not above and data.get(attribute, 0) >= threshold:
			H.remove_edge(u, v)
	return H

def local_degree_sparsifier(G: nx.Graph, 
							target_ratio: float) -> nx.Graph:
	"""
	Sparsifies graph based on local node degrees.

	Args:
		G (nx.Graph): Input graph
		target_ratio (float): Target ratio of edges to keep

	Returns:
		nx.Graph: Sparsified graph
	"""
	H = nx.Graph()
	H.add_nodes_from(G.nodes())
	
	edges = sorted(G.edges(data=True), 
				   key=lambda x: min(G.degree(x[0]), G.degree(x[1])),
				   reverse=True)
	
	target_edges = int(G.number_of_edges() * target_ratio)
	H.add_edges_from(edges[:target_edges])
	
	return H

def random_edge_sparsifier(G: nx.Graph, 
						   target_ratio: float, 
						   seed: int = 42) -> nx.Graph:
	"""
	Randomly sparsifies edges.

	Args:
		G (nx.Graph): Input graph
		target_ratio (float): Target ratio of edges to keep
		seed (int): Random seed for reproducibility

	Returns:
		nx.Graph: Sparsified graph
	"""
	np.random.seed(seed)
	H = G.copy()
	
	num_to_remove = int(G.number_of_edges() * (1 - target_ratio))
	edges_to_remove = list(G.edges())
	np.random.shuffle(edges_to_remove)
	edges_to_remove = edges_to_remove[:num_to_remove]
	
	H.remove_edges_from(edges_to_remove)
	return H

def simmelian_sparsifier(G: nx.Graph, 
						 max_rank: int = 5) -> nx.Graph:
	"""
	Implements Simmelian backbone sparsification.

	Args:
		G (nx.Graph): Input graph
		max_rank (int): Maximum rank considered for overlap calculation

	Returns:
		nx.Graph: Sparsified graph
	"""
	def simmelian_strength(u, v):
		u_neighbors = set(G.neighbors(u))
		v_neighbors = set(G.neighbors(v))
		common_neighbors = u_neighbors.intersection(v_neighbors)
		return len(common_neighbors)
	
	H = nx.Graph()
	H.add_nodes_from(G.nodes())
	
	for u in G.nodes():
		neighbors = sorted(G.neighbors(u), 
						   key=lambda x: simmelian_strength(u, x),
						   reverse=True)
		H.add_edges_from((u, v) for v in neighbors[:max_rank])
	
	return H

def disparity_filter(G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
	"""
	Implements the disparity filter technique as described in 
	Serrano et al. (2009) PNAS paper.

	Args:
		G (nx.Graph): Input weighted graph
		alpha (float): Significance level for the filter

	Returns:
		nx.Graph: Filtered graph
	"""

	H = nx.Graph()
	H.add_nodes_from(G.nodes())

	for u in G.nodes():
		k = G.degree(u)
		if k > 1:
			strength = sum(G[u][v].get('weight', 1) for v in G[u])
			for v in G[u]:
				weight = G[u][v].get('weight', 1)
				p_ij = weight / strength
				alpha_ij = 1 - (k - 1) * scipy.integrate.quad(lambda x: (1 - x)**(k-2), 0, p_ij)[0]
				if alpha_ij < alpha:
					H.add_edge(u, v, weight=weight)

	return H

def overlapping_trees(G: nx.Graph, num_trees: int = 3) -> nx.Graph:
	"""
	Implements the Overlapping Trees network reduction technique as described in
	Carmi et al. (2008) arXiv:0812.3227.

	This method creates a reduced network by combining multiple spanning trees.

	Args:
		G (nx.Graph): Input weighted graph
		num_trees (int): Number of spanning trees to generate and combine

	Returns:
		nx.Graph: Reduced graph
	"""
	# Initialize the reduced graph
	H = nx.Graph()
	H.add_nodes_from(G.nodes())

	# Generate multiple minimum spanning trees
	for _ in range(num_trees):
		# Generate random weights for this iteration
		for (u, v, d) in G.edges(data=True):
			d['random_weight'] = np.random.random()

		# Compute the minimum spanning tree using the random weights
		T = nx.minimum_spanning_tree(G, weight='random_weight')

		# Add the edges from this tree to the reduced graph
		H.add_edges_from(T.edges(data=True))

	# Transfer original edge weights to the reduced graph
	for (u, v, d) in H.edges(data=True):
		d['weight'] = G[u][v].get('weight', 1)

	return H

def k_core_decomposition(G: nx.Graph, k: int = None) -> nx.Graph:
	"""
	Implements the k-core decomposition network reduction technique.

	This method creates a reduced network by recursively removing nodes with degree less than k,
	until no such nodes remain. If k is not specified, it returns the main core (largest k-core).

	Args:
		G (nx.Graph): Input graph
		k (int, optional): The order of the core. If not specified, returns the main core.

	Returns:
		nx.Graph: Reduced graph (k-core subgraph)

	References:
		Batagelj, V., & Zaversnik, M. (2003). An O(m) Algorithm for Cores Decomposition of Networks.
		https://arxiv.org/abs/cs.DS/0310049
	"""
	# Compute core numbers for all nodes
	core_numbers = nx.core_number(G)

	if k is None:
		# If k is not specified, use the maximum core number (main core)
		k = max(core_numbers.values())

	# Create a subgraph with nodes having core number >= k
	H = G.subgraph([n for n, cn in core_numbers.items() if cn >= k])

	return H
