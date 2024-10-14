import os
import pickle
import networkx as nx
import sys
import pandas as pd
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.benchmark import Benchmark
from benchmark.networkIndicators import NetworkIndicators
from net_filtering.filter import Filter

class Analysis:
    def get_all_networks(self) -> list[dict]:
        """
        Get all graphs (as NetworkX objects) from the specified directory, including their file names and whether they are weighted or unweighted.

        Returns:
        list[dict]: A list of dictionaries containing graphs, file names, and their category (weighted or unweighted).
        """
        # Base directory where the graphs are stored
        base_directory = 'data/simulated_nets'
        graphs = []

        # Traverse through the base directory to find subdirectories
        for subdir in os.listdir(base_directory):
            subdir_path = os.path.join(base_directory, subdir)
            if not os.path.isdir(subdir_path):
                continue

            # Type can be weighted or unweighted
            for type in os.listdir(subdir_path):
                target_path = os.path.join(subdir_path, type)
                if not os.path.isdir(target_path):
                    continue

                # Traverse through the target directory to find pickle files
                for file in os.listdir(target_path):
                    if file.endswith('.pickle'):
                        filepath = os.path.join(target_path, file)
                        try:
                            with open(filepath, 'rb') as f:
                                graph = pickle.load(f)
                                graphs.append({
                                'graph': graph,
                                'weighted': type == 'weighted',
                                'filename': file
                            })
                        except Exception as e:
                            print(f"Error loading graph from {filepath}: {e}")
        
        return graphs
    
    def generate_results(self):
        """
         Generate results based on combinations of networks, filtering methods, benchmarks, and indicators.
        """
        # Create an instance of Filter, Benchmark and NetworkIndicators classes
        filter_instance = Filter()
        benchmark_instance = Benchmark()
        netIndicators_instance = NetworkIndicators()

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

        # List of benchmark techniques
        benchmark_funcs = [
            benchmark_instance.bench_net2net_filtering,
            benchmark_instance.bench_noise_filtering,
            benchmark_instance.bench_structural_noise_filtering
        ]

        # List of network indicators techniques
        net_indicators_methods = [
            netIndicators_instance.calculate_information_retention,
            netIndicators_instance.calculate_jaccard_distance,
            #netIndicators_instance.calculate_network_similarity
        ]

        # List of networks
        networks = self.get_all_networks()

        # Creating combinations between networks, filters, benchmarks, and indicators
        combinations = list(itertools.product(networks, filtering_funcs, benchmark_funcs,
                                            net_indicators_methods))
        
        # Filtering invalid combinations
        filters_that_require_weights = [filter_instance.mst, filter_instance.threshold]

        valid_combinations = [
            combo for combo in combinations if combo[0]['weighted'] or combo[1] not in filters_that_require_weights
        ]

        # Creating a DataFrame with these combinations
        df = pd.DataFrame(valid_combinations, columns=['network', 'filter', 'benchmark', 'indicator'])

        # Function to apply each combination
        def apply_benchmark(row):
            graph = row['network']['graph']
            filter_func = row['filter']
            benchmark_func = row['benchmark']
            indicator_func = row['indicator']

            print(row['network']['filename'])

            print(row.name)

            # Treat filter_func based on the specified logic
            filter = filter_func
            if filter_func == filter_instance.threshold:
                filter = lambda G: filter_func(G, threshold=0.5)
            elif filter_func in [filter_instance.local_degree_sparsifier, filter_instance.random_edge_sparsifier]:
                filter = lambda G: filter_func(G, target_ratio=0.5)
            
            result = benchmark_func(graph, filter, indicator_func)
            return result

        # Applying the function to each row in the DataFrame
        df['result'] = df.apply(apply_benchmark, axis=1)

        # Creating a mapping of function objects to their names
        filter_names = {filter_instance.mst: 'MST', 
                        filter_instance.threshold: 'Threshold',
                        filter_instance.local_degree_sparsifier: 'Local Degree Sparsifier',
                        filter_instance.random_edge_sparsifier: 'Random Edge Sparsifier',
                        filter_instance.simmelian_sparsifier: 'Simmelian Sparsifier',
                        filter_instance.disparity_filter: 'Disparity Filter',
                        filter_instance.overlapping_trees: 'Overlapping Trees',
                        filter_instance.k_core_decomposition: 'K-Core Decomposition'}

        benchmark_names = {benchmark_instance.bench_net2net_filtering: 'Net2Net Filtering',
                        benchmark_instance.bench_noise_filtering: 'Noise Filtering',
                        benchmark_instance.bench_structural_noise_filtering: 'Structural Noise Filtering'}

        indicator_names = {netIndicators_instance.calculate_information_retention: 'Information Retention',
                        netIndicators_instance.calculate_jaccard_distance: 'Jaccard Distance',
                        netIndicators_instance.calculate_network_similarity: 'Network Similarity'}

        # Replace function objects with their string names
        df['filter'] = df['filter'].map(filter_names)
        df['benchmark'] = df['benchmark'].map(benchmark_names)
        df['indicator'] = df['indicator'].map(indicator_names)

        # Adding columns for filename and weighted based on the values in the 'network' dictionary
        df['filename'] = df['network'].apply(lambda x: x['filename'].replace('.pickle', ''))
        df['weighted'] = df['network'].apply(lambda x: x['weighted'])

        # Dropping the 'network' column
        df.drop(columns=['network'], inplace=True)

        df.to_csv('results/results.csv')

analysis = Analysis()

analysis.generate_results()