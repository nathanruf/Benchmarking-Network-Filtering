import networkx as nx
import os
import pickle
import logging
import random
import math
from networkx.generators.community import LFR_benchmark_graph
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory where every simulated network is saved (one place to change it).
SIMULATED_NETS_DIR = 'data/simulated_nets'

# Ensure the data/simulated directory exists
os.makedirs(SIMULATED_NETS_DIR, exist_ok=True)

# Define an Enum for Graph Types
class GraphType(Enum):
	RANDOM = 'random'
	GRID = 'grid'
	BARABASI_ALBERT = 'barabasi_albert'
	LFR_BENCHMARK = 'lfr_benchmark'
	WATTS_STROGATZ = 'watts_strogatz'

def add_random_weights(G, seed=None):
	"""
	Add random weights between 0 and 1 to all edges in the graph.

	Uses a local, seeded random.Random instance (instead of the global
	random module) so that, given the same graph and the same seed, the
	assigned weights are always the same.

	Args:
		G (networkx.Graph): The input graph.
		seed (int, optional): Seed controlling the weight values. Default is None
			(non-deterministic).

	Returns:
		networkx.Graph: The graph with random weights added to edges.
	"""
	rng = random.Random(seed)
	for (u, v) in G.edges():
		G[u][v]['weight'] = rng.uniform(0, 1)
	return G

def generate_random_graph(n, p, seed=42, weighted=False):
	"""
	Generate a random graph using the Erdős-Rényi model.

	Args:
		n (int): Number of nodes.
		p (float): Probability of edge creation.
		seed (int, optional): Random seed for reproducibility. Default is 42.
		weighted (bool): If True, add random weights to edges.

	Returns:
		networkx.Graph: Generated random graph.
	"""
	G = nx.erdos_renyi_graph(n, p, seed=seed)
	if weighted:
		G = add_random_weights(G, seed=seed)
	logger.info(f"Generated {'weighted ' if weighted else ''}random graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_grid_graph(m, n, periodic=False, weighted=False, seed=42):
	"""
	Generate a grid graph.

	Grid topology is fully determined by (m, n, periodic) -- there is no
	randomness to seed for the topology itself. `seed` only controls the
	edge weights (when weighted=True), so regenerating the same grid always
	yields the same weights too.

	Args:
		m (int): Number of rows.
		n (int): Number of columns.
		periodic (bool): Whether to make the grid periodic.
		weighted (bool): If True, add random weights to edges.
		seed (int, optional): Seed for the (deterministic) weight assignment. Default is 42.

	Returns:
		networkx.Graph: Generated grid graph.
	"""
	if periodic:
		G = nx.grid_2d_graph(m, n, periodic=True)
	else:
		G = nx.grid_2d_graph(m, n)
	if weighted:
		G = add_random_weights(G, seed=seed)
	logger.info(f"Generated {'weighted ' if weighted else ''}{'periodic ' if periodic else ''}grid graph with {m}x{n} nodes")
	return G

def generate_barabasi_albert_graph(n, m, seed=42, weighted=False):
	"""
	Generate a Barabási-Albert preferential attachment graph.

	Args:
		n (int): Number of nodes.
		m (int): Number of edges to attach from a new node to existing nodes.
		seed (int, optional): Random seed for reproducibility. Default is 42.
		weighted (bool): If True, add random weights to edges.

	Returns:
		networkx.Graph: Generated Barabási-Albert graph.
	"""
	G = nx.barabasi_albert_graph(n, m, seed=seed)
	if weighted:
		G = add_random_weights(G, seed=seed)
	logger.info(f"Generated {'weighted ' if weighted else ''}Barabási-Albert graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_lfr_benchmark_graph(n, tau1, tau2, mu, average_degree=10, min_community=5, max_community=20, max_degree=20, seed=42, weighted=False):
	"""
	Generate a graph with community structure using the LFR benchmark model.

	Args:
		n (int): Number of nodes.
		tau1 (float): Power law exponent for the degree distribution.
		tau2 (float): Power law exponent for the community size distribution.
		mu (float): Mixing parameter.
		average_degree (int): Average degree of nodes.
		min_community (int): Minimum community size.
		seed (int, optional): Random seed for reproducibility. Default is 42.
		weighted (bool): If True, add random weights to edges.

	Returns:
		networkx.Graph: Generated LFR benchmark graph.
	"""
	G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=average_degree,
							min_community=min_community, max_community=max_community, max_degree=max_degree, seed=seed)
	G = nx.Graph(G)
	G.remove_edges_from(nx.selfloop_edges(G))

	if weighted:
		G = add_random_weights(G, seed=seed)
	logger.info(f"Generated {'weighted ' if weighted else ''}LFR benchmark graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_watts_strogatz_graph(n, k, p, seed=42, weighted=False):
	"""
	Generate a Watts-Strogatz small-world graph.

	Args:
		n (int): Number of nodes.
		k (int): Each node is connected to k nearest neighbors in ring topology.
		p (float): Probability of rewiring each edge.
		seed (int, optional): Random seed for reproducibility. Default is 42.
		weighted (bool): If True, add random weights to edges.

	Returns:
		networkx.Graph: Generated Watts-Strogatz small-world graph.
	"""
	G = nx.watts_strogatz_graph(n, k, p, seed=seed)
	if weighted:
		G = add_random_weights(G, seed=seed)
	logger.info(f"Generated {'weighted ' if weighted else ''}Watts-Strogatz small-world graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def save_graph(G, directory, filename):
	"""
	Save a graph to a file.

	Args:
		G (networkx.Graph): Graph to save.
		directory (str): Directory path to save the graph.
		filename (str): Name of the file to save the graph.
	"""
	os.makedirs(directory, exist_ok=True)

	filepath = f'{directory}{filename}'

	with open(filepath, 'wb') as f:
		pickle.dump(G, f)
	logger.info(f"Saved graph to {filepath}")

def generate_and_save_graph(generator_func, graph_type, filename, override=False, **kwargs):
	"""
	Generate a graph and save it if it doesn't exist or if override is True.

	Every graph (weighted or not) is saved under the same path -- filters that
	don't care about weight simply ignore the 'weight' edge attribute, so there
	is no need to keep separate weighted/unweighted copies of the same topology.

	Args:
		generator_func (function): Function to generate the graph.
		graph_type (GraphType): Type of the graph (e.g., 'random', 'grid').
		filename (str): Name of the file to save the graph.
		override (bool): Whether to override existing files.
		**kwargs: Additional arguments for the generator function.

	Returns:
		bool: True if a new graph was generated and saved, False otherwise.
	"""
	directory = f'{SIMULATED_NETS_DIR}/{graph_type.name.lower()}/'
	filepath = f'{directory}{filename}'

	if not override and os.path.exists(filepath):
		logger.info(f"File {filepath} already exists. Skipping generation.")
		return False

	G = generator_func(**kwargs)
	save_graph(G, directory, filename)
	return True

def main(override=False):
	"""
	Generate and save example graphs.

	Args:
		override (bool): Whether to override existing files. Default is False.
	"""

	# Grid graphs are deterministic given (rows, columns, periodic) -- there is no
	# randomness to vary between "replicates", so only one graph is generated per n
	# instead of 100 identical copies. `seed=n` just keeps its (weighted) edge
	# weights reproducible across regenerations.
	for n in range(100, 1100, 100):
		rows = int(math.sqrt(n))
		columns = n // rows
		generate_and_save_graph(
			generate_grid_graph, GraphType.GRID, f'grid_graph{n}.pickle', override,
			m=rows, n=columns, periodic=True, weighted=True, seed=n,
		)

	for i in range(0, 100):
		for n in range(100, 1100, 100):
			mu = 0.1
			min_community = int(0.05*n)
			average_degree = int(min_community/2.5)
			max_degree = int(2.5 * average_degree)
			max_community = int(0.2*n)

			# Generate and save example graphs. `seed=i` makes each of the 100
			# replicates an independent draw from the model instead of the same
			# graph saved under 100 different filenames.
			generate_and_save_graph(generate_random_graph, GraphType.RANDOM, f'{i}random_graph{n}.pickle', override, n=n, p=0.1, seed=i, weighted=True)
			generate_and_save_graph(generate_barabasi_albert_graph, GraphType.BARABASI_ALBERT, f'{i}barabasi_albert_graph{n}.pickle', override, n=n, m=2, seed=i, weighted=True)
			generate_and_save_graph(generate_lfr_benchmark_graph, GraphType.LFR_BENCHMARK, f'{i}lfr_benchmark_graph{n}.pickle', override, n=n, tau1=2.5, tau2=1.5,
						    mu=mu, min_community=min_community, average_degree = average_degree, max_degree=max_degree, max_community=max_community, seed=i, weighted=True)
			generate_and_save_graph(generate_watts_strogatz_graph, GraphType.WATTS_STROGATZ, f'{i}watts_strogatz_graph{n}.pickle', override, n=n, k=4, p=0.1, seed=i, weighted=True)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Generate simulated networks")
	parser.add_argument("--override", action="store_true", help="Override existing pickle files")
	args = parser.parse_args()

	main(override=args.override)
