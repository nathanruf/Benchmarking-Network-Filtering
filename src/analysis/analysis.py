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
from pandarallel import pandarallel

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

        # List of noise levels
        noise_levels = [
            None,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5
        ]

        # List of networks
        networks = self.get_all_networks()

        # Creating combinations between networks, filters, benchmarks, and indicators
        combinations = list(itertools.product(networks, filtering_funcs, benchmark_funcs, noise_levels))
        
        # Filtering invalid combinations
        filters_that_require_weights = [filter_instance.mst, filter_instance.threshold]

        # If the benchmark function is 'bench_net2net_filtering', noise_level must be None
        # For other benchmark functions, noise_level must not be None
        valid_combinations = []
        for combo in combinations:
            network, filter_func, benchmark_func, noise_level = combo
            # Check if the combination is valid based on the filter and benchmark
            if (network['weighted'] or filter_func not in filters_that_require_weights) and \
            (benchmark_func != benchmark_instance.bench_net2net_filtering or noise_level is None) and \
            (benchmark_func == benchmark_instance.bench_net2net_filtering or noise_level is not None):
                valid_combinations.append(combo)

        # Creating a DataFrame with these combinations
        df = pd.DataFrame(valid_combinations, columns=['network', 'filter', 'benchmark', 'noise_level'])
        
        result_columns = ['information_retention', 'jaccard', 'degree_assortativity_original', 
                      'degree_assortativity_filtered', 'average_clustering_original', 'average_clustering_filtered',
                      'average_shortest_path_length_original', 'average_shortest_path_length_filtered', 'density_original', 
                      'density_filtered', 'average_degree_connectivity_original', 'average_degree_connectivity_filtered',
                      'transitivity_original', 'transitivity_filtered', 'true_positive', 'true_negative', 'false_positive', 
                      'false_negative', 'precision', 'recall', 'f1_score', 'RMSE']

        for column in result_columns:
            df[column] = None  # Initialize each result column with None
        
        # Function to apply each combination
        def apply_benchmark(row):

            graph = row['network']['graph']
            filter_func = row['filter']
            benchmark_func = row['benchmark']
            noise = row['noise_level']

            print(row.name)

            # Treat filter_func based on the specified logic
            filter = filter_func
            if filter_func.__name__ == filter_instance.threshold.__name__:
                filter = lambda G: filter_func(G, threshold=0.5)
            elif filter_func.__name__ in [filter_instance.local_degree_sparsifier.__name__, filter_instance.random_edge_sparsifier.__name__]:
                filter = lambda G: filter_func(G, target_ratio=0.5)

            noisy_benchmark_funcs = [benchmark_instance.bench_noise_filtering.__name__, benchmark_instance.bench_structural_noise_filtering.__name__]
            benchmark = benchmark_func
            if benchmark_func.__name__ in noisy_benchmark_funcs:
                benchmark = lambda G, F, I: benchmark_func(G, F, I, noise_level = noise)

            row['information_retention'] = benchmark(graph, filter, netIndicators_instance.calculate_information_retention)
            row['jaccard'] = benchmark(graph, filter, netIndicators_instance.calculate_jaccard_similarity)
            common_metrics = benchmark(graph, filter, netIndicators_instance.common_metrics)

            for key, value in common_metrics.items():
                row[key] = value

            predictive_filtering_metrics = benchmark(graph, filter, netIndicators_instance.predictive_filtering_metrics)

            for key, value in predictive_filtering_metrics.items():
                row[key] = value
                
            return row
        
        # Initialize pandarallel
        pandarallel.initialize()

        # Applying the function to each row in the DataFrame
        df = df.parallel_apply(apply_benchmark, axis=1)
        
        # Replace function objects with their string names
        df['filter'] = df['filter'].map(lambda x: x.__name__)
        df['benchmark'] = df['benchmark'].map(lambda x: x.__name__)

        # Adding columns for filename and weighted based on the values in the 'network' dictionary
        df['filename'] = df['network'].apply(lambda x: x['filename'].replace('.pickle', ''))
        df['weighted'] = df['network'].apply(lambda x: x['weighted'])

        # Dropping the 'network' column
        df.drop(columns=['network'], inplace=True)

        df.to_csv('results/results.csv')

if __name__ == '__main__':
    analysis = Analysis()

    analysis.generate_results()