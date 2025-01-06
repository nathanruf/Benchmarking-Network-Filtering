import os
import pickle
import networkx as nx
import sys
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np
import re
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

        filtering_funcs = [
                filter_instance.mst,
                # PMFG and TMFG methods are taking too long, consider optimizing or commenting it out for now
                #filter_instance.pmfg,
                #filter_instance.tmfg,
                filter_instance.threshold,
                filter_instance.local_degree_sparsifier,
                filter_instance.random_edge_sparsifier,
                filter_instance.simmelian_sparsifier,
                #Disparity filter not working as expected
                #filter_instance.disparity_filter,
                filter_instance.overlapping_trees,
                filter_instance.k_core_decomposition,
                #Verify coarse_graining_espectral results 
                #filter_instance.coarse_graining_espectral
            ]

            # List of benchmark techniques
        benchmark_funcs = [
                benchmark_instance.bench_noise_filtering,
                benchmark_instance.bench_structural_noise_filtering
            ]

            # List of noise levels
        noise_levels = [
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
        filters_that_require_weights = [filter_instance.mst, filter_instance.pmfg, filter_instance.tmfg, filter_instance.threshold]
        filters_that_not_require_weights = [filter_instance.coarse_graining_espectral, filter_instance.k_core_decomposition,
                                            filter_instance.local_degree_sparsifier, filter_instance.overlapping_trees,
                                            filter_instance.random_edge_sparsifier, filter_instance.simmelian_sparsifier]

        # If the benchmark function is 'bench_net2net_filtering', noise_level must be None
        # For other benchmark functions, noise_level must not be None
        valid_combinations = []
        for combo in combinations:
            network, filter_func, _, _ = combo
            # Check if the combination is valid based on the filter and benchmark
            if ( (network['weighted'] and filter_func in filters_that_require_weights) or 
                (not network['weighted'] and filter_func in filters_that_not_require_weights) ):
                valid_combinations.append(combo)

        # Creating a DataFrame with these combinations
        df = pd.DataFrame(valid_combinations, columns=['network', 'filter', 'benchmark', 'noise_level'])
        
        result_columns = ['information_retention', 'jaccard', 'average_degree_original', 'average_degree_filtered',
                          'average_clustering_original', 'average_clustering_filtered',
                          'average_path_length_original', 'average_path_length_filtered',
                          'diameter_original', 'diameter_filtered',
                          #'average_betweenness_original', 'average_betweenness_filtered',
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
            RANDOM = 'Random'
            GRID = 'Grid'
            BARABASI_ALBERT = 'Barabasi_albert'
            WATTS_STROGATZ = 'Watts_strogatz'
            BIOLOGICAL = 'Biological'
            ECONOMIC = 'Economic'
            INFORMATION = 'Informational'
            SOCIAL = 'Social'
            TECHNOLOGICAL = 'Technological'
            TRANSPORTATION = 'Transportation'
            REAL_NETS = 'Real_nets'

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
                        benchmark_path = os.path.join(graph_path, benchmark)
                        os.makedirs(benchmark_path, exist_ok=True)

        df_mean = pd.read_csv('results/resultsMean.csv')
        df_std = pd.read_csv('results/resultsStd.csv')

        benchmarks = df_mean['benchmark'].unique()

        base_path = 'results/graphics'

        create_directories(base_path, benchmarks)

        for benchmark in benchmarks:
            benchmark_results_mean = df_mean[df_mean['benchmark'] == benchmark]
            benchmark_results_std = df_std[df_std['benchmark'] == benchmark]

            for graph_type in GraphType:
                # Filter data by graph type
                type_results_mean = benchmark_results_mean[benchmark_results_mean['class'] == graph_type.value]
                type_results_std = benchmark_results_std[benchmark_results_std['class'] == graph_type.value]

                # Class 1: Noise level x Jaccard for each benchmark
                class_1_path = os.path.join(base_path, 'class_1', graph_type.value, benchmark)
                os.makedirs(class_1_path, exist_ok=True)

                for filter in type_results_mean['filter'].unique():
                    df_filter_mean = type_results_mean[type_results_mean['filter'] == filter].drop_duplicates(subset=['noise_level', 'jaccard'])
                    df_filter_std = type_results_std[type_results_std['filter'] == filter].drop_duplicates(subset=['noise_level', 'jaccard'])

                    plt.errorbar(
                        df_filter_mean['noise_level'],  # x
                        df_filter_mean['jaccard'],  # y
                        yerr=df_filter_std['jaccard'],  # Erro (desvio padrão)
                        fmt='-o',  # Formato dos pontos
                        label=f'{filter}'
                    )

                plt.title(f'Jaccard Score by Noise Level - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel('Jaccard Score')
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                plt.ylim(0,1)
                plt.savefig(os.path.join(class_1_path, f'Jaccard_{graph_type}.png'))
                plt.close()

                # Class 2: Precision, Recall, and F1-score - Separate plots
                class_2_path = os.path.join(base_path, 'class_2', graph_type.value, benchmark)
                os.makedirs(class_2_path, exist_ok=True)

                for metric in ['precision', 'recall', 'f1_score']:
                    for filter in type_results_mean['filter'].unique():
                        df_filter_mean = type_results_mean[type_results_mean['filter'] == filter]
                        df_filter_std = type_results_std[type_results_std['filter'] == filter]

                        plt.errorbar(
                            df_filter_mean['noise_level'],  # x
                            df_filter_mean[metric],  # y
                            yerr=df_filter_std[metric],  # Erro (desvio padrão)
                            fmt='-o',  # Formato dos pontos
                            label=f'{filter}'
                        )

                    plt.title(f'{metric.capitalize()} by Noise Level - {benchmark}')
                    plt.xlabel('Noise Level')
                    plt.ylabel(f'{metric.capitalize()} Score')
                    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                    plt.ylim(0,1)
                    plt.savefig(os.path.join(class_2_path, f'{metric}_{graph_type}.png'))
                    plt.close()

                # Class 3: cosine distance for metrics
                class_3_path = os.path.join(base_path, 'class_3', graph_type.value, benchmark)
                os.makedirs(class_3_path, exist_ok=True)

                original_metrics = ['average_degree_original', 'average_clustering_original', 'average_closeness_original',
                                    'degree_assortativity_original', 'degree_variance_original']
                filtered_metrics = ['average_degree_filtered', 'average_clustering_filtered', 'average_closeness_filtered',
                                    'degree_assortativity_filtered', 'degree_variance_filtered']
                
                for filter in type_results_mean['filter'].unique():
                    df_filter_mean = type_results_mean[type_results_mean['filter'] == filter]
                    df_filter_std = type_results_std[type_results_std['filter'] == filter]

                    original_metrics_std = df_filter_std[original_metrics]
                    filtered_metrics_std = df_filter_std[filtered_metrics]

                    partial_derivate_original_mean = df_filter_mean.apply(lambda row: (np.dot(row[filtered_metrics], np.linalg.norm(row[original_metrics])) -
                                            np.dot(row[original_metrics] ,(np.dot(row[original_metrics], row[filtered_metrics]) / np.linalg.norm(row[original_metrics])))
                                            ) / np.dot( (np.linalg.norm(row[original_metrics]) ** 2), np.linalg.norm(row[filtered_metrics]))
                                            , axis = 1)
                    
                    partial_derivate_filtered_mean = df_filter_mean.apply(lambda row: (np.dot(row[original_metrics], np.linalg.norm(row[filtered_metrics])) -
                                            np.dot(row[filtered_metrics] ,(np.dot(row[filtered_metrics], row[original_metrics]) / np.linalg.norm(row[filtered_metrics])))
                                            ) / np.dot( (np.linalg.norm(row[filtered_metrics]) ** 2), np.linalg.norm(row[original_metrics]))
                                            , axis = 1)
                    
                    cosine_distance_mean = df_filter_mean.apply(lambda row: cosine(row[original_metrics], row[filtered_metrics]), axis = 1)

                    cosine_distance_std = []

                    for key, row_original_std in original_metrics_std.iterrows():
                        cosine_distance_std.append(
                            np.sqrt(
                            (np.dot(partial_derivate_original_mean[key], row_original_std) ** 2) + 
                            (np.dot(partial_derivate_filtered_mean[key], filtered_metrics_std.loc[key]) ** 2)
                        )
                        )

                    plt.errorbar(
                        df_filter_mean['noise_level'],  # x
                        cosine_distance_mean,  # y
                        yerr=cosine_distance_std,  # Erro (desvio padrão)
                        fmt='-o',  # Formato dos pontos
                        label=f'{filter}'
                    )
                    
                plt.title(f'Cosine distance by Noise Level - {benchmark}')
                plt.xlabel('Noise Level')
                plt.ylabel(f'{metric.capitalize()} Score')
                plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                plt.savefig(os.path.join(class_3_path, f'cosine_distance_{graph_type}.png'))
                plt.close()

                # Class 4: Variation of metrics by percentage
                class_4_path = os.path.join(base_path, 'class_4', graph_type.value, benchmark)
                os.makedirs(class_4_path, exist_ok=True)

                metrics = ['average_degree', 'average_clustering', 'average_path_length',
                            'diameter', 'average_closeness',
                            'global_efficiency', 'degree_assortativity', 'density',
                            'transitivity', 'degree_variance', 'maximum_degree']

                for metric_base in metrics:
                    plt.figure()
                    
                    for filter in type_results_mean['filter'].unique():
                        filter_data_mean = type_results_mean[type_results_mean['filter'] == filter]
                        filter_data_std = type_results_std[type_results_std['filter'] == filter]
                        
                        # Calculate percentage variation between original and filtered metrics for each filter
                        original_metric_mean = filter_data_mean[f'{metric_base}_original']
                        filtered_metric_mean = filter_data_mean[f'{metric_base}_filtered']
                        variation_mean = (filtered_metric_mean - original_metric_mean) / original_metric_mean

                        original_metric_std = filter_data_std[f'{metric_base}_original']
                        filtered_metric_std = filter_data_std[f'{metric_base}_filtered']
                        variation_std = np.sqrt(
                                                        ((filtered_metric_std/original_metric_mean) ** 2) + 
                                                        (((filtered_metric_mean  * original_metric_std) / (original_metric_mean ** 2)) ** 2) 
                                                    )
                        
                        # Plotting the percentage variation for the metric with each filter
                        plt.errorbar(
                            df_filter_mean['noise_level'],  # x
                            variation_mean,  # y
                            yerr=variation_std,  # Erro (desvio padrão)
                            fmt='-o',  # Formato dos pontos
                            label=f'{filter}'
                        )
                    
                    # Set up titles and labels for the plot
                    plt.title(f'{metric_base} Variation by Percentage - {benchmark}')
                    plt.xlabel('Noise Level')
                    plt.ylabel('Percentage Change')
                    
                    # Adjust legend position and save the plot
                    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                    plt.savefig(os.path.join(class_4_path, f'{metric_base}_variation_{graph_type}.png'))
                    plt.close()

                # Class 5: cosine distance for metrics
                class_5_path = os.path.join(base_path, 'class_5', graph_type.value, benchmark)
                os.makedirs(class_5_path, exist_ok=True)

                metrics = ['average_degree',
                            'average_clustering',
                            'average_path_length',
                            'diameter',
                            'average_closeness',
                            'global_efficiency',
                            'degree_assortativity',
                            'density',
                            'transitivity',
                            'degree_variance',
                            'maximum_degree']
                
                for metric in metrics:
                    for filter in type_results_mean['filter'].unique():
                        df_filter_mean = type_results_mean[type_results_mean['filter'] == filter]
                        df_filter_std = type_results_std[type_results_std['filter'] == filter]

                        df_filter_std = pd.concat([
                            pd.DataFrame({'noise_level': [0.0], 
                                        f'{metric}_filtered': [df_filter_std[f'{metric}_original'].iloc[0]],
                                        'filter': [filter]}),
                            df_filter_std
                        ], ignore_index=True)

                        df_filter_mean = pd.concat([
                            pd.DataFrame({'noise_level': [0.0], 
                                        f'{metric}_filtered': [df_filter_mean[f'{metric}_original'].iloc[0]],
                                        'filter': [filter]}),
                            df_filter_mean
                        ], ignore_index=True)
                        
                        plt.errorbar(
                            df_filter_mean['noise_level'],  # x
                            df_filter_mean[f'{metric}_filtered'],  # y
                            yerr=df_filter_std[f'{metric}_filtered'],  # Erro (desvio padrão)
                            fmt='-o',  # Formato dos pontos
                            label=f'{filter}'
                        )

                    plt.title(f'{metric.capitalize()} by Noise Level - {benchmark}')
                    plt.xlabel('Noise Level')
                    plt.ylabel(f'{metric.capitalize()} Score')
                    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  # Adjust legend position
                    plt.savefig(os.path.join(class_5_path, f'{metric}_{graph_type}.png'))
                    plt.close()

    def generate_class_results(self):
        graph_types = ['Random', 'Grid', 'Barabasi_albert', 'Watts_strogatz']

        simulated_nets = pd.read_csv('results/simulatedNetsResults.csv', index_col=0)
        simulated_nets['class'] = simulated_nets['filename'].apply(lambda filename: next(graph_type for graph_type in graph_types if graph_type.lower() in filename))
        simulated_nets['size'] = simulated_nets['filename'].apply(lambda filename: int(filename.split('_')[-1][5:]))
        simulated_nets.drop('filename', axis = 1, inplace=True)

        real_nets = pd.read_csv('results/realNetsResults.csv')
        real_nets['filename'] = real_nets['filename'].str.replace('_real_nets', '')
        real_nets_info = pd.read_csv('data/real_nets/ICON_info.csv')
        real_nets = real_nets.merge(real_nets_info[['network_name', 'networkDomain','number_nodes']], 
                             left_on='filename', right_on='network_name', 
                             how='left')

        real_nets['class'] = real_nets['networkDomain']
        real_nets['size'] = real_nets['number_nodes']
        real_nets.drop(['networkDomain', 'network_name', 'filename', 'number_nodes'], axis=1, inplace=True)

        real_nets_group = real_nets.copy()
        real_nets_group['class'] = 'Real_nets'
        real_nets = pd.concat([real_nets, real_nets_group])

        group_columns = ['class', 'noise_level', 'weighted', 'benchmark', 'filter']

        nets = pd.concat([real_nets, simulated_nets])

        mean_res = nets.copy()

        for column in mean_res.columns:
            if column in group_columns + ['size']:
                continue
            mean_res[column] = mean_res[column] * mean_res['size']
            
        mean_res = mean_res.groupby(group_columns).sum(numeric_only=True)
        
        for column in mean_res.columns:
            if column in group_columns + ['size']:
                continue
            mean_res[column] = mean_res[column] / mean_res['size']

        mean_res.drop('size', axis=1, inplace=True)

        std_res = mean_res.copy().rename(columns = lambda col: 'mean_' + col if col not in group_columns else col)

        std_res = pd.merge(nets, std_res, on = group_columns, how='inner')

        for column in std_res.columns:
            if column in group_columns + ['size'] or 'mean_' in column:
                continue
            std_res[column] = std_res['size'] * ((std_res[column] - std_res[f'mean_{column}']) ** 2)

        std_res.drop([column for column in std_res.columns if 'mean_' in column], axis=1, inplace=True)
        std_res = std_res.groupby(group_columns).sum(numeric_only=True)

        for column in std_res.columns:
            if column in group_columns + ['size']:
                continue
            std_res[column] = (std_res[column] / std_res['size']) ** 0.5

        std_res.drop('size', axis=1, inplace=True)

        mean_res.to_csv('results/resultsMean.csv')
        std_res.to_csv('results/resultsStd.csv')

if __name__ == '__main__':
    analysis = Analysis()
    analysis.generate_graphics()