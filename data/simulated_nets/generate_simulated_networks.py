import networkx as nx
import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the data/simulated directory exists
os.makedirs('data/simulated_nets', exist_ok=True)

def generate_random_graph(n, p, seed=None):
	"""
	Generate a random graph using the Erdős-Rényi model.

	Args:
		n (int): Number of nodes.
		p (float): Probability of edge creation.
		seed (int, optional): Random seed for reproducibility.

	Returns:
		networkx.Graph: Generated random graph.
	"""
	G = nx.erdos_renyi_graph(n, p, seed=seed)
	logger.info(f"Generated random graph with {n} nodes and {G.number_of_edges()} edges")
	return G

def generate_grid_graph(m, n, periodic=False):
	"""
	Generate a grid graph.

	Args:
		m (int): Number of rows.
		n (int): Number of columns.
		periodic (bool): Whether to make the grid periodic.

	Returns:
		networkx.Graph: Generated grid graph.
	"""
	if periodic:
		G = nx.grid_2d_graph(m, n, periodic=True)
	else:
		G = nx.grid_2d_graph(m, n)
	logger.info(f"Generated {'periodic ' if periodic else ''}grid graph with {m}x{n} nodes")
	return G

def generate_barabasi_albert_graph(n, m, seed=None):
	"""
	Generate a Barabási-Albert preferential attachment graph.

	Args:
		n (int): Number of nodes.
		m (int): Number of edges to attach from a new node to existing nodes.
		seed (int, optional): Random seed for reproducibility.

	Returns:
		networkx.Graph: Generated Barabási-Albert graph.
	"""
	G = nx.barabasi_albert_graph(n, m, seed=seed)
	logger.info(f"Generated Barabási-Albert graph with {n} nodes and {G.number_of_edges()} edges")
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
	random_graph = generate_random_graph(100, 0.1, seed=42)
	save_graph(random_graph, 'random_graph.pickle')

	grid_graph = generate_grid_graph(10, 10, periodic=True)
	save_graph(grid_graph, 'grid_graph.pickle')

	ba_graph = generate_barabasi_albert_graph(100, 2, seed=42)
	save_graph(ba_graph, 'barabasi_albert_graph.pickle')