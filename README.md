
##  Benchmarking Network Filtering

This repository implements a framework for benchmarking various network filtering techniques. It also provides real and simulated networks and it implements common network filtering techniques for benchmarking purposes. See original research proposal [HERE](https://docs.google.com/document/d/1d4vKYAfspwY5npEHu1PBNh5hgWPiw_A6idKMB5Vh7UE/edit).

## Project Structure

- **data/:** Includes sample and real networks for benchmarking
  - **real_nets/:** Networks extracted from Index of Complex Networks (ICON)
  - **simulated_nets/:** Random networks
- **src/:** Contains the source code for the benchmarking framework
  - **benchmark/:** Benchmarking utilities and core functionality
  - **net_filtering/:** Implementation of various network filtering techniques
- **results/:** Directory to store benchmark results and performance metrics

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nathanruf/Benchmarking-Network-Filtering.git
   ```

2. Navigate to the project directory:
   ```
   cd Benchmarking-Network-Filtering
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run a benchmark:

1. Ensure you have the necessary dataset in the `data/` directory.
2. Use the `src/benchmark/bench_noise_filtering.py` script to benchmark a specific filter:

   ```python
   from src.benchmark.bench_noise_filtering import bench_noise_filtering
   from src.net_filtering.filter import Filter
   import networkx as nx

   # Load your network
   G = pickle.load("/data/simulated_nets/random_graph.pickle")

   # Create a filter instance
   filter_instance = Filter()

   # Run the benchmark
   result = bench_noise_filtering(G, filter_instance.mst)
   print(f"Result: {result")
   ```


## Benchmarking Process

1. The framework adds noise to the input network using the `add_noise_to_network` function.
2. It then applies the specified filtering technique to the noisy network.
3. The performance is evaluated by comparing the filtered network to the original network.

## Available Data

The `/data` directory contains both real and simulated network datasets for benchmarking purposes.

### Real Networks

The `/data/real_nets/` directory contains networks extracted from ICON (Index of Complex Networks). These networks represent various real-world systems and phenomena across different domains. See networks metadata [here](https://docs.google.com/spreadsheets/d/1DCSPqD3cLDKZ00QC7NjZpjgnE33coCXwigjxTY5NhYc/edit?usp=sharing).

1. Full Data: `real_net.pickle`

Corpus of 550 real-world networks drawn from the Index of Complex Networks (ICON) used in this [PNAS paper](https://github.com/Aghasemian/OptimalLinkPrediction). This corpus spans a variety of sizes and structures, with 23% social, 23% economic, 32% biological, 12% technological, 3% information, and 7% transportation graphs.

2. Sample Data: `real_net_sample.pickle`

A sample of 4 hand-picked networks to run quick benchmarking; network_index = [447, 133, 122, 80]. Refer to [ICON network metadata](https://docs.google.com/spreadsheets/d/1DCSPqD3cLDKZ00QC7NjZpjgnE33coCXwigjxTY5NhYc/edit?usp=sharing) to view details of those sample networks.


### Simulated Networks

The `/data/simulated_nets/` directory contains artificially generated network datasets. These include:

1. Random Graphs (Erdős-Rényi model)
   - Unweighted: `random_graph.pickle`
   - Weighted: `random_graph_weighted.pickle`

2. Grid Graphs
   - Unweighted: `grid_graph.pickle`
   - Weighted: `grid_graph_weighted.pickle`

3. Barabási-Albert Graphs (Scale-free networks)
   - Unweighted: `barabasi_albert_graph.pickle`
   - Weighted: `barabasi_albert_graph_weighted.pickle`

4. LFR Benchmark Graphs (with community structure)
   - Unweighted: `lfr_benchmark_graph.pickle`
   - Weighted: `lfr_benchmark_graph_weighted.pickle`

5. Watts-Strogatz Graphs (Small-world networks)
   - Unweighted: `watts_strogatz_graph.pickle`
   - Weighted: `watts_strogatz_graph_weighted.pickle`

These simulated networks provide a diverse set of graph structures and properties, allowing for comprehensive benchmarking of filtering techniques across various network types.

To generate new simulated networks or modify existing ones, refer to the `data/simulated_nets/generate_simulated_networks.py` script.

## Available Network Filtering Techniques

The `src/net_filtering/filter.py` file contains various network filtering and graph sparsification techniques. Here's an overview of the available methods:

1. Minimum Spanning Tree (MST)
   - Method: `mst(graph)`
   - Computes the Minimum Spanning Tree of a graph, keeping the minimum set of edges that connect all nodes with the lowest total edge weight.

2. Planar Maximally Filtered Graph (PMFG)
   - Method: `pmfg(graph)`
   - Constructs a planar graph that maximizes the sum of edge weights while maintaining planarity.

3. Global Threshold Filter
   - Method: `threshold(graph, threshold)`
   - Applies a global threshold to edge weights, removing edges below the specified threshold.

4. Local Degree Sparsifier
   - Method: `local_degree_sparsifier(G, target_ratio)`
   - Sparsifies the graph based on local node degrees, keeping a target ratio of edges.

5. Random Edge Sparsifier
   - Method: `random_edge_sparsifier(G, target_ratio, seed=42)`
   - Randomly removes edges to achieve a target sparsity ratio.

6. Simmelian Backbone Sparsifier
   - Method: `simmelian_sparsifier(G, max_rank=5)`
   - Implements Simmelian backbone sparsification, focusing on strongly embedded edges.

7. Disparity Filter
   - Method: `disparity_filter(G, alpha=0.05)`
   - Implements the disparity filter technique as described in Serrano et al. (2009) PNAS paper.

8. Overlapping Trees
   - Method: `overlapping_trees(G, num_trees=3)`
   - Creates a reduced network by combining multiple spanning trees.

9. K-Core Decomposition
   - Method: `k_core_decomposition(G, k=None)`
   - Implements k-core decomposition, recursively removing nodes with degree less than k.

These filtering techniques can be applied to both weighted and unweighted graphs, providing a diverse set of approaches for network reduction and noise filtering. Each method has its own strengths and is suitable for different types of network structures and analysis goals.

### Adding New Filters

To add a new filtering technique:

1. Implement your filter in the `src/net_filtering/filter.py` file.
2. Ensure your filter function takes a networkx Graph as input and returns a filtered Graph.
3. Use the `bench_noise_filtering` function to benchmark your new filter.






