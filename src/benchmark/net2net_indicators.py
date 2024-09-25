"""
This module provides functions for calculating various network indicators
using NetworkX graphs. These indicators can be used to analyze and compare
different network structures.

Functions:
- calculate_information_retention: Calculates the Information Retention score
  between an original and a filtered network.
- calculate_network_similarity: Calculates the similarity between two networks
  based on common network metrics.
"""

import networkx as nx
import numpy as np
from scipy.stats import entropy
from typing import Union, List
from scipy.spatial.distance import cosine

def calculate_information_retention(original_graph: Union[nx.Graph, nx.DiGraph], 
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

def calculate_network_similarity(network1: Union[nx.Graph, nx.DiGraph], 
								network2: Union[nx.Graph, nx.DiGraph]) -> float:
	"""
	Calculate the similarity between two networks based on common network metrics.

	This function computes various network metrics for both input networks,
	creates vectors of these metrics, and then calculates the cosine similarity
	between these vectors.

	Parameters:
	network1 (Union[nx.Graph, nx.DiGraph]): The first network
	network2 (Union[nx.Graph, nx.DiGraph]): The second network

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
		# Calculate metrics for both networks
		metrics1 = calculate_metrics(network1)
		metrics2 = calculate_metrics(network2)

		# Calculate cosine similarity
		similarity = 1 - cosine(metrics1, metrics2)  # Convert distance to similarity

		return similarity

	except Exception as e:
		print(f"Error calculating network similarity: {str(e)}")
		return None
	

def calculate_jaccard_distance(network1: Union[nx.Graph, nx.DiGraph], network2: Union[nx.Graph, nx.DiGraph]) -> float:
	"""
	Calculate the Jaccard distance between two networks based on their edge sets.

	The Jaccard distance is defined as 1 minus the Jaccard similarity coefficient.
	It measures the dissimilarity between two sets, in this case, the edge sets of the networks.

	Parameters:
	network1 (Union[nx.Graph, nx.DiGraph]): The first network
	network2 (Union[nx.Graph, nx.DiGraph]): The second network

	Returns:
	float: Jaccard distance between the two networks (lower is more similar)
	"""
	try:
		# Get edge sets for both networks
		edges1 = set(network1.edges())
		edges2 = set(network2.edges())

		# Calculate Jaccard similarity coefficient
		intersection = len(edges1.intersection(edges2))
		union = len(edges1.union(edges2))

		jaccard_similarity = intersection / union if union > 0 else 0

		# Calculate Jaccard distance
		jaccard_distance = 1 - jaccard_similarity

		return jaccard_distance

	except Exception as e:
		print(f"Error calculating Jaccard distance: {str(e)}")
		return None
