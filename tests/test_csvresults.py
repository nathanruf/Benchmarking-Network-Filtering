import pandas as pd

class TestCsvResults:
    def test_density_matches_formula(self):
        real = pd.read_csv('results/realNetsResults.csv')
        nets_info = pd.read_csv('data/real_nets/ICON_info.csv')

        real['filename'] = real['filename'].str.replace('_real_nets', '')

        real = real[real['benchmark'] != 'bench_structural_noise_filtering']

        real = pd.merge(real, nets_info, left_on = 'filename', right_on = 'network_name')

        assert (real['density_original'].round(2) == ((2 * real['number_edges']) / (real['number_nodes'] * (real['number_nodes'] - 1))).round(2) ).all(), 'Density should match with the formula'
    
    def test_clustering_index_is_greater_than_0_1_for_ws(self):
        simulated = pd.read_csv('results/simulatedNetsResults.csv')

        ws_nets = simulated[simulated['filename'].str.contains('watts')]

        ws_nets = ws_nets[ws_nets['benchmark'] != 'bench_structural_noise_filtering']

        assert (ws_nets['average_clustering_original'] > 0.1).all(), 'Clustering index should be greater than 0.1'

    def test_filtered_density_is_smaller_than_original(self):
        simulated = pd.read_csv('results/simulatedNetsResults.csv')
        real = pd.read_csv('results/realNetsResults.csv')

        simulated = simulated[(simulated['benchmark'] == 'bench_net2net_filtering') & (simulated['filter'] != 'k_core_decomposition')]
        real = real[(real['benchmark'] == 'bench_net2net_filtering') & (real['filter'] != 'k_core_decomposition')]

        assert (simulated['density_original'] >= simulated['density_filtered']).all(), 'Original density should be greater than filtered - simulated nets'
        assert (real['density_original'] >= real['density_filtered']).all(), 'Original density should be greater than filtered - real nets'

    def test_metrics_are_numeric(self):
        simulated = pd.read_csv('results/simulatedNetsResults.csv')
        real = pd.read_csv('results/realNetsResults.csv')

        simulated = simulated.drop(['filter', 'benchmark', 'noise_level', 'average_path_length_original', 'average_path_length_filtered',
                        'diameter_original', 'diameter_filtered', 'filename', 'weighted'], axis=1)
        real = real.drop(['filter', 'benchmark', 'noise_level', 'average_path_length_original', 'average_path_length_filtered',
                        'diameter_original', 'diameter_filtered', 'filename', 'weighted'], axis=1)
        
        assert simulated.select_dtypes(include = 'number').shape[1] == simulated.shape[1], 'Simulated results contain non-numeric values'
        assert real.select_dtypes(include = 'number').shape[1] == real.shape[1], 'Real results contain non-numeric values'

    def test_original_density_greater_than_zero(self):
        simulated = pd.read_csv('results/simulatedNetsResults.csv')
        real = pd.read_csv('results/realNetsResults.csv')

        assert (simulated['density_original'] > 0).all(), 'Original density for simulated nets is not greater than zero'
        assert (real['density_original'] > 0).all(), 'Original density for real nets is not greater than zero'

    def test_filtered_density_greater_than_zero(self):
        simulated = pd.read_csv('results/simulatedNetsResults.csv')
        real = pd.read_csv('results/realNetsResults.csv')

        assert (simulated['density_filtered'] > 0).all(), 'Filtered density for simulated nets is not greater than zero'
        assert (real['density_filtered'] > 0).all(), 'Filtered density for real nets is not greater than zero'