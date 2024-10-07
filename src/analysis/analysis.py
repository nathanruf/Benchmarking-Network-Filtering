import os
import pickle
import networkx as nx
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.benchmark import Benchmark
from net_filtering.filter import Filter

class Analysis:
    def get_all_networks(self, weighted: bool) -> list[nx.Graph]:
        """
        Get all graphs (as NetworkX objects) from the specified directory based on whether they are weighted or unweighted.

        Args:
            weighted (bool): If True, fetch weighted graphs; otherwise, fetch unweighted graphs.

        Returns:
            list[nx.Graph]: A list of graphs loaded from the pickle files.
        """
        # Base directory where the graphs are stored
        base_directory = 'data/simulated_nets'
        subfolder = 'weighted' if weighted else 'unweighted'
        graphs = []

        # Traverse through the base directory to find subdirectories
        for subdir in os.listdir(base_directory):
            subdir_path = os.path.join(base_directory, subdir)
            # Check if it's a directory
            if os.path.isdir(subdir_path):
                # Look for 'weighted' or 'unweighted' inside this subdirectory
                target_path = os.path.join(subdir_path, subfolder)
                if os.path.exists(target_path) and os.path.isdir(target_path):
                    # Traverse through the target directory to find pickle files
                    for file in os.listdir(target_path):
                        if file.endswith('.pickle'):
                            filepath = os.path.join(target_path, file)
                            try:
                                with open(filepath, 'rb') as f:
                                    graph = pickle.load(f)
                                    graphs.append(graph)
                            except Exception as e:
                                print(f"Error loading graph from {filepath}: {e}")
        
        return graphs
    
    def generate_results(self, weighted: bool):
        """
         Generate results by applying various filtering techniques and benchmark methods to graphs.

        Args:
            weighted (bool): If True, fetch weighted graphs; otherwise, fetch unweighted graphs.
        """
        # Create an instance of Filter and Benchmark classes
        filter_instance = Filter()
        benchmark_instance = Benchmark()

        # List of filtering techniques
        filtering_funcs = [
            filter_instance.mst,
            # PMFG method is taking too long, consider optimizing or commenting it out for now
            #filter_instance.pmfg,
            filter_instance.threshold,
            filter_instance.local_degree_sparsifier,
            filter_instance.random_edge_sparsifier,
            filter_instance.simmelian_sparsifier,
            filter_instance.disparity_filter,
            filter_instance.overlapping_trees,
            filter_instance.k_core_decomposition
        ]

        # Remove filters for weighted graphs if weighted is false.
        if not weighted:
            filtering_funcs.remove(filter_instance.mst)
            filtering_funcs.remove(filter_instance.pmfg)
            filtering_funcs.remove(filter_instance.threshold)

        # List of benchmark techniques
        benchmark_funcs = [
            benchmark_instance.bench_noise_filtering,
            benchmark_instance.bench_structural_noise_filtering,
            benchmark_instance.calculate_information_retention,
            benchmark_instance.calculate_network_similarity,
            benchmark_instance.calculate_jaccard_distance
        ]

        # List of networks
        networks = self.get_all_networks(weighted)

        for net in networks:
            for filter in filtering_funcs:
                if filter == filter_instance.threshold:
                    filter_func = lambda G: filter(G, threshold=0.5)
                elif filter in [filter_instance.local_degree_sparsifier, filter_instance.random_edge_sparsifier]:
                    filter_func = lambda G: filter(G, target_ratio=0.5)
                else:
                    filter_func = filter
                for bench in benchmark_funcs:
                    print(bench(net, filter_func))


analysis = Analysis()

analysis.generate_results(True)