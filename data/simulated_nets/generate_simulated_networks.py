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

# Ensure the data/simulated directory exists
os.makedirs('data/simulated_nets', exist_ok=True)

# Define an Enum for Graph Types
class GraphType(Enum):
    RANDOM = 'random'
    GRID = 'grid'
    BARABASI_ALBERT = 'barabasi_albert'
    LFR_BENCHMARK = 'lfr_benchmark'
    WATTS_STROGATZ = 'watts_strogatz'

def add_random_weights(G):
	"""
	Add random weights between 0 and 1 to all edges in the graph.

	Args:
		G (networkx.Graph): The input graph.

	Returns:
		networkx.Graph: The graph with random weights added to edges.
	"""
	for (u, v) in G.edges():
		G[u][v]['weight'] = random.uniform(0, 1)
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
		G = add_random_weights(G)
	logger.info(f"Generated {'weighted ' if weighted else ''}random graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_grid_graph(m, n, periodic=False, weighted=False):
	"""
	Generate a grid graph.

	Args:
		m (int): Number of rows.
		n (int): Number of columns.
		periodic (bool): Whether to make the grid periodic.
		weighted (bool): If True, add random weights to edges.

	Returns:
		networkx.Graph: Generated grid graph.
	"""
	if periodic:
		G = nx.grid_2d_graph(m, n, periodic=True)
	else:
		G = nx.grid_2d_graph(m, n)
	if weighted:
		G = add_random_weights(G)
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
		G = add_random_weights(G)
	logger.info(f"Generated {'weighted ' if weighted else ''}Barabási-Albert graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_lfr_benchmark_graph(n, tau1, tau2, mu, average_degree=10, min_community=20, seed=42, weighted=False):
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
							min_community=min_community, seed=seed)
	if weighted:
		G = add_random_weights(G)
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
		G = add_random_weights(G)
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

	Args:
		generator_func (function): Function to generate the graph.
		graph_type (GraphType): Type of the graph (e.g., 'random', 'grid').
		filename (str): Name of the file to save the graph.
		override (bool): Whether to override existing files.
		**kwargs: Additional arguments for the generator function.

	Returns:
		bool: True if a new graph was generated and saved, False otherwise.
	"""
	weighted = kwargs.get('weighted', False)
	
	directory = f'data/simulated_nets/{graph_type.name.lower()}/'

	# Define filepath based on whether it's weighted or unweighted
	if weighted:
		directory += 'weighted/'
	else:
		directory += 'unweighted/'

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

	for n in range(100, 1100, 100):
		rows = int(math.sqrt(n))
		columns = n // rows

		# Generate and save example graphs
		generate_and_save_graph(generate_random_graph, GraphType.RANDOM, f'random_graph{n}.pickle', override, n=n, p=0.1, weighted=True)
		generate_and_save_graph(generate_grid_graph, GraphType.GRID, f'grid_graph{n}.pickle', override, m=rows, n=columns, periodic=True, weighted=True)
		generate_and_save_graph(generate_barabasi_albert_graph, GraphType.BARABASI_ALBERT, f'barabasi_albert_graph{n}.pickle', override, n=n, m=2, weighted=True)
		#lfr_benchmark_graph is generating an error
		#generate_and_save_graph(generate_lfr_benchmark_graph, GraphType.LFR_BENCHMARK, f'lfr_benchmark_graph{n}.pickle', override, n=n, tau1=2.5, tau2=1.5, mu=0.1, weighted=True)
		generate_and_save_graph(generate_watts_strogatz_graph, GraphType.WATTS_STROGATZ, f'watts_strogatz_graph{n}.pickle', override, n=n, k=4, p=0.1, weighted=True)

		# Generate unweighted versions for comparison
		generate_and_save_graph(generate_random_graph, GraphType.RANDOM, f'random_graph{n}.pickle', override, n=n, p=0.1, weighted=False)
		generate_and_save_graph(generate_grid_graph, GraphType.GRID, f'grid_graph{n}.pickle', override, m=rows, n=columns, periodic=True, weighted=False)
		generate_and_save_graph(generate_barabasi_albert_graph, GraphType.BARABASI_ALBERT, f'barabasi_albert_graph{n}.pickle', override, n=n, m=2, weighted=False)
		#lfr_benchmark_graph is generating an error
		#generate_and_save_graph(generate_lfr_benchmark_graph, GraphType.LFR_BENCHMARK, f'lfr_benchmark_graph{n}.pickle', override, n=n, tau1=2.5, tau2=1.5, mu=0.1, weighted=False)
		generate_and_save_graph(generate_watts_strogatz_graph, GraphType.WATTS_STROGATZ, f'watts_strogatz_graph{n}.pickle', override, n=n, k=4, p=0.1, weighted=False)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Generate simulated networks")
	parser.add_argument("--override", action="store_true", help="Override existing pickle files")
	args = parser.parse_args()

	main(override=args.override)