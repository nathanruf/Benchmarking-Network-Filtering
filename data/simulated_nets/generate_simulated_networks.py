import networkx as nx
import os
import pickle
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the data/simulated directory exists
os.makedirs('data/simulated_nets', exist_ok=True)

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

def save_graph(G, filename):
	"""
	Save a graph to a file.

	Args:
		G (networkx.Graph): Graph to save.
		filename (str): Name of the file to save the graph.
	"""
	with open(f'data/simulated_nets/{filename}', 'wb') as f:
		pickle.dump(G, f)
	logger.info(f"Saved graph to data/simulated_nets/{filename}")

if __name__ == "__main__":
	# Generate and save example graphs
	random_graph = generate_random_graph(100, 0.1, weighted=True)
	save_graph(random_graph, 'random_graph_weighted.pickle')

	grid_graph = generate_grid_graph(10, 10, periodic=True, weighted=True)
	save_graph(grid_graph, 'grid_graph_weighted.pickle')

	ba_graph = generate_barabasi_albert_graph(100, 2, weighted=True)
	save_graph(ba_graph, 'barabasi_albert_graph_weighted.pickle')

	# Generate unweighted versions for comparison
	random_graph_unweighted = generate_random_graph(100, 0.1, weighted=False)
	save_graph(random_graph_unweighted, 'random_graph.pickle')

	grid_graph_unweighted = generate_grid_graph(10, 10, periodic=True, weighted=False)
	save_graph(grid_graph_unweighted, 'grid_graph.pickle')

	ba_graph_unweighted = generate_barabasi_albert_graph(100, 2, weighted=False)
	save_graph(ba_graph_unweighted, 'barabasi_albert_graph.pickle')