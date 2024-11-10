import os
import pickle
import networkx as nx
import sys
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from concurrent.futures import ProcessPoolExecutor
from benchmark.benchmark import Benchmark
from benchmark.networkIndicators import NetworkIndicators
from net_filtering.filter import Filter

class Analysis:
    def get_all_networks(self, real_net:bool) -> list[dict]:
        """
        Get all graphs (as NetworkX objects) from the specified directory, including their file names and whether they are weighted or unweighted.

        Returns:
        list[dict]: A list of dictionaries containing graphs, file names, and their category (weighted or unweighted).
        """
        if real_net:
            filepath = "data/real_nets/real_net.pickle"
            graphs = None
            try:
                with open(filepath, 'rb') as f:
                    graphs = pickle.load(f)
            except Exception as e:
                print(f"Error loading graph from {filepath}: {e}")

            return graphs

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
    
    def apply_benchmark(self, row):
            netIndicators_instance = NetworkIndicators()

            graph = row['network']['graph']
            filter_func = row['filter']
            benchmark_func = row['benchmark']
            noise = row['noise_level']

            print(row.name)

            # Treat filter_func based on the specified logic
            filter = filter_func
            if filter_func.__name__ == Filter.threshold.__name__:
                filter = lambda G: filter_func(G, threshold=0.5)
            elif filter_func.__name__ in [Filter.local_degree_sparsifier.__name__, Filter.random_edge_sparsifier.__name__]:
                filter = lambda G: filter_func(G, target_ratio=0.5)

            noisy_benchmark_funcs = [Benchmark.bench_noise_filtering.__name__, Benchmark.bench_structural_noise_filtering.__name__]
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
    
    def generate_results(self, real_net:bool):
        """
         Generate results based on combinations of networks, filtering methods, benchmarks, and indicators.
        """
        start_time = time.time()

        # Create an instance of Filter, Benchmark and NetworkIndicators classes
        filter_instance = Filter()
        benchmark_instance = Benchmark()

        filtering_funcs = [
            filter_instance.simmelian_sparsifier
        ]

        # List of benchmark techniques
        benchmark_funcs = [
            benchmark_instance.bench_structural_noise_filtering
        ]

        # List of noise levels
        noise_levels = [
            0.5
        ]

        # List of networks
        networks = self.get_all_networks(real_net)

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

        # Use ProcessPoolExecutor to parallelize
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(self.apply_benchmark, [row for index, row in df.iterrows()]))

        # Transformar os resultados em um DataFrame novamente
        df = pd.DataFrame(results)

        # Replace function objects with their string names
        df['filter'] = df['filter'].map(lambda x: x.__name__)
        df['benchmark'] = df['benchmark'].map(lambda x: x.__name__)

        # Adding columns for filename and weighted based on the values in the 'network' dictionary
        df['filename'] = df['network'].apply(lambda x: x['filename'].replace('.pickle', ''))
        df['weighted'] = df['network'].apply(lambda x: x['weighted'])

        # Dropping the 'network' column
        df.drop(columns=['network'], inplace=True)

        df.to_csv('results/realNetsResults.csv', index=False)

        end_time = time.time()  # Marca o tempo de fim
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(f"Time to generate the results: {formatted_time}")

    def generate_graphics(self) -> None:
        df = pd.read_csv('results/results.csv')

        df = df[df['filename'].str.contains('1000')]

        results = {}

        for benchmark in df['benchmark'].unique():
            benchmark_results = df[df['benchmark'] == benchmark]

            for filename in df['filename'].unique():
                file_benchmark_results = benchmark_results[benchmark_results['filename'] == filename]

                for filter in file_benchmark_results['filter'].unique():
                    df_filter = file_benchmark_results[file_benchmark_results['filter'] == filter]

                    df_filter = df_filter.drop_duplicates(subset = ('noise_level', 'jaccard'))

                    plt.plot(df_filter['noise_level'], df_filter['jaccard'], label=f'{filter}')

                plt.title('Gráfico classe 1')
                plt.xlabel('Noise level')
                plt.ylabel('Jaccard score')
                plt.legend()
                plt.savefig(f'results/{filename}_{benchmark}.png')
                plt.close('all')
                bolo = plt.gcf
                #clear the plt

if __name__ == '__main__':
    analysis = Analysis()

    analysis.generate_results(True)