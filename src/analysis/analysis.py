import os
import pickle
import networkx as nx
import sys
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cosine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

            indicator_functions = [
                netIndicators_instance.calculate_information_retention,
                netIndicators_instance.calculate_jaccard_similarity,
                netIndicators_instance.common_metrics,
                netIndicators_instance.predictive_filtering_metrics
            ]

            row['information_retention'], row['jaccard'], common_metrics, predictive_filtering_metrics = benchmark(graph, filter, indicator_functions)

            for key, value in common_metrics.items():
                row[key] = value

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

        # List of filtering techniques
        filtering_funcs = [
            filter_instance.mst,
            # PMFG method is taking too long, consider optimizing or commenting it out for now
            #filter_instance.pmfg,
            filter_instance.tmfg,
            filter_instance.threshold,
            filter_instance.local_degree_sparsifier,
            filter_instance.random_edge_sparsifier,
            filter_instance.simmelian_sparsifier,
            #Disparity filter not working as expected
            #filter_instance.disparity_filter,
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
        networks = self.get_all_networks(real_net)

        # Creating combinations between networks, filters, benchmarks, and indicators
        combinations = list(itertools.product(networks, filtering_funcs, benchmark_funcs, noise_levels))
        
        # Filtering invalid combinations
        filters_that_require_weights = [filter_instance.mst, filter_instance.tmfg, filter_instance.threshold]

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
        
        result_columns = ['information_retention', 'jaccard', 'average_degree_original', 'average_degree_filtered',
                          'average_clustering_original', 'average_clustering_filtered',
                          'average_path_length_original', 'average_path_length_filtered',
                          'diameter_original', 'diameter_filtered',
                          'average_betweenness_original', 'average_betweenness_filtered',
                          'average_closeness_original', 'average_closeness_filtered',
                          'global_efficiency_original', 'global_efficiency_filtered',
                          'degree_assortativity_original', 'degree_assortativity_filtered',
                          'density_original', 'density_filtered',
                          'transitivity_original', 'transitivity_filtered',
                          'degree_variance_original', 'degree_variance_filtered',
                          'maximum_degree_original', 'maximum_degree_filtered', 'true_positive', 'true_negative',
                          'false_positive', 'false_negative', 'precision', 'recall', 'f1_score', 'RMSE']

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

        filename = 'realNetsResults.csv' if real_net else 'simulatedNetsResults.csv'
        df.to_csv(f'results/{filename}', index=False)

        end_time = time.time()  # Marca o tempo de fim
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(f"Time to generate the results: {formatted_time}")

    def generate_graphics(self) -> None:
        """
        Generate and save graphs in four classes based on different metrics.
        Filters data for graphs of size 1000 and saves in the appropriate directories.
        """
        class GraphType(Enum):
            RANDOM = 'random'
            GRID = 'grid'
            BARABASI_ALBERT = 'barabasi_albert'
            WATTS_STROGATZ = 'watts_strogatz'
            REAL_NETS = 'real_nets'

        def create_directories(base_path, benchmarks):
            """
            Create the directory structure for storing the graphs based on class, graph type, 
            benchmark type, and whether they are weighted or unweighted.
            """
            for class_num in range(1, 6):
                class_path = os.path.join(base_path, f'class_{class_num}')
                os.makedirs(class_path, exist_ok=True)

                for graph_type in GraphType:
                    graph_path = os.path.join(class_path, graph_type.value)
                    os.makedirs(graph_path, exist_ok=True)

                    for benchmark in benchmarks:
                        if benchmark == 'bench_net2net_filtering':
                            continue
                        benchmark_path = os.path.join(graph_path, benchmark)
                        os.makedirs(os.path.join(benchmark_path, 'weighted'), exist_ok=True)
                        os.makedirs(os.path.join(benchmark_path, 'unweighted'), exist_ok=True)

        df = pd.read_csv('results/simulatedNetsResults.csv')

        df = df[df['filename'].str.contains('1000')]
        df = pd.concat([df, pd.read_csv('results/realNetsSample.csv')], ignore_index=True)
        benchmarks = df['benchmark'].unique()

        base_path = 'results/graphics'

        create_directories(base_path, benchmarks)

        for benchmark in benchmarks:
            if benchmark == 'bench_net2net_filtering':
                continue

            benchmark_results = df[df['benchmark'] == benchmark]

            for filename in df['filename'].unique():
                file_benchmark_results = benchmark_results[benchmark_results['filename'] == filename]

                for graph_type in GraphType:
                    # Filter data by graph type
                    type_results = file_benchmark_results[file_benchmark_results['filename'].str.contains(graph_type.value)]

                    for weighted in [True, False]:
                        weight_str = 'weighted' if weighted else 'unweighted'
                        weight_results = type_results[type_results['weighted'] == weighted]

                        if weight_results.empty:
                            continue

                        # Class 1: Noise level x Jaccard for each benchmark
                        class_1_path = os.path.join(base_path, 'class_1', graph_type.value, benchmark, weight_str)
                        os.makedirs(class_1_path, exist_ok=True)

                        for filter in weight_results['filter'].unique():
                            df_filter = weight_results[weight_results['filter'] == filter].drop_duplicates(subset=['noise_level', 'jaccard'])

                            plt.plot(df_filter['noise_level'], df_filter['jaccard'], label=filter)

                        plt.title(f'Jaccard Score by Noise Level - {benchmark}')
                        plt.xlabel('Noise Level')
                        plt.ylabel('Jaccard Score')
                        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                        plt.savefig(os.path.join(class_1_path, f'Jaccard_{filename}.png'))
                        plt.close()

                        # Class 2: Precision, Recall, and F1-score - Separate plots
                        class_2_path = os.path.join(base_path, 'class_2', graph_type.value, benchmark, weight_str)
                        os.makedirs(class_2_path, exist_ok=True)

                        for metric in ['precision', 'recall', 'f1_score']:
                            for filter in weight_results['filter'].unique():
                                df_filter = weight_results[weight_results['filter'] == filter]
                                plt.plot(df_filter['noise_level'], df_filter[metric], label=filter)

                            plt.title(f'{metric.capitalize()} by Noise Level - {benchmark}')
                            plt.xlabel('Noise Level')
                            plt.ylabel(f'{metric.capitalize()} Score')
                            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                            plt.savefig(os.path.join(class_2_path, f'{metric}_{filename}.png'))
                            plt.close()

                        # Class 3: cosine distance for metrics
                        class_3_path = os.path.join(base_path, 'class_3', graph_type.value, benchmark, weight_str)
                        os.makedirs(class_3_path, exist_ok=True)

                        original_metrics = ['average_degree_original', 'average_clustering_original', 'average_path_length_original',
                                            'diameter_original', 'average_betweenness_original', 'average_closeness_original',
                                            'global_efficiency_original', 'degree_assortativity_original', 'density_original',
                                            'transitivity_original', 'degree_variance_original', 'maximum_degree_original']
                        filtered_metrics = ['average_degree_filtered', 'average_clustering_filtered', 'average_path_length_filtered',
                                            'diameter_filtered', 'average_betweenness_filtered', 'average_closeness_filtered',
                                            'global_efficiency_filtered', 'degree_assortativity_filtered', 'density_filtered',
                                            'transitivity_filtered', 'degree_variance_filtered', 'maximum_degree_filtered']
                        
                        for filter in weight_results['filter'].unique():
                            df_filter = weight_results[weight_results['filter'] == filter]
                            cosine_distance = df_filter.apply(lambda row : cosine(row[original_metrics], row[filtered_metrics]), axis = 1)
                            plt.plot(df_filter['noise_level'], cosine_distance, label=filter)

                        plt.title(f'Cosine distance by Noise Level - {benchmark}')
                        plt.xlabel('Noise Level')
                        plt.ylabel(f'{metric.capitalize()} Score')
                        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                        plt.savefig(os.path.join(class_3_path, f'cosine_distance_{filename}.png'))
                        plt.close()

                        # Class 4: Variation of metrics by percentage
                        class_4_path = os.path.join(base_path, 'class_4', graph_type.value, benchmark, weight_str)
                        os.makedirs(class_4_path, exist_ok=True)

                        metrics = ['average_degree', 'average_clustering', 'average_path_length',
                                    'diameter', 'average_betweenness', 'average_closeness',
                                    'global_efficiency', 'degree_assortativity', 'density',
                                    'transitivity', 'degree_variance', 'maximum_degree']

                        for metric_base in metrics:
                            plt.figure()
                            
                            for filter_type in weight_results['filter'].unique():
                                filter_data = weight_results[weight_results['filter'] == filter_type]
                                
                                # Calculate percentage variation between original and filtered metrics for each filter
                                original_metric = filter_data[f'{metric_base}_original']
                                filtered_metric = filter_data[f'{metric_base}_filtered']
                                percentage_variation = ((filtered_metric - original_metric) / original_metric) * 100
                                
                                # Plotting the percentage variation for the metric with each filter
                                plt.plot(filter_data['noise_level'], percentage_variation, label=f'{filter_type}')
                            
                            # Set up titles and labels for the plot
                            plt.title(f'{metric_base} Variation by Percentage - {benchmark}')
                            plt.xlabel('Noise Level')
                            plt.ylabel('Percentage Change')
                            
                            # Adjust legend position and save the plot
                            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                            plt.savefig(os.path.join(class_4_path, f'{metric_base}_variation_{filename}.png'))
                            plt.close()

                        # Class 5: cosine distance for metrics
                        class_5_path = os.path.join(base_path, 'class_5', graph_type.value, benchmark, weight_str)
                        os.makedirs(class_5_path, exist_ok=True)

                        metrics = ['average_degree',
                                   'average_clustering',
                                   'average_path_length',
                                   'diameter',
                                   'average_betweenness',
                                   'average_closeness',
                                   'global_efficiency',
                                   'degree_assortativity',
                                   'density',
                                   'transitivity',
                                   'degree_variance',
                                   'maximum_degree']
                        
                        for metric in metrics:
                            for filter in weight_results['filter'].unique():
                                df_filter = weight_results[weight_results['filter'] == filter]

                                df_filter = pd.concat([
                                    pd.DataFrame({'noise_level': [0.0], 
                                                f'{metric}_filtered': [df_filter[f'{metric}_original'].iloc[0]],
                                                'filter': [filter]}),
                                    df_filter
                                ], ignore_index=True)
                                
                                plt.plot(df_filter['noise_level'], df_filter[f'{metric}_filtered'], label=filter)

                            plt.title(f'{metric.capitalize()} by Noise Level - {benchmark}')
                            plt.xlabel('Noise Level')
                            plt.ylabel(f'{metric.capitalize()} Score')
                            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                            plt.savefig(os.path.join(class_5_path, f'{metric}_{filename}.png'))
                            plt.close()

if __name__ == '__main__':
    analysis = Analysis()

    analysis.generate_results(False)