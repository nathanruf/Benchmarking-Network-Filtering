
##  Benchmarking Network Filtering

This repository implements a framework for benchmarking various network filtering techniques. It also provides real and simulated networks and it implements common network filtering techniques for benchmarking purposes.

## Project Structure

- **data/:** Includes sample datasets and network traces for testing
  - **real_nets/:** Networks extracted from ICON
  - **simulated_nets/:** Simulated network datasets
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


## Available Data

The `/data` directory contains both real and simulated network datasets for benchmarking purposes:

### Real Networks

The `/data/real_nets/` directory contains networks extracted from ICON (Index of Complex Networks). These networks represent various real-world systems and phenomena across different domains. See networks metadata [here](https://docs.google.com/spreadsheets/d/1DCSPqD3cLDKZ00QC7NjZpjgnE33coCXwigjxTY5NhYc/edit?usp=sharing).

1. Full Data: `real_net.pickle`

Corpus of 550 real-world networks drawn from the Index of Complex Networks (ICON) used in this [PNAS paper](https://github.com/Aghasemian/OptimalLinkPrediction). This corpus spans a variety of sizes and structures, with 23% social, 23% economic, 32% biological, 12% technological, 3% information, and 7% transportation graphs (Fig. S1 of the paper).

2. Sample Data: `real_net_sample.pickle`

A sample of 4 hand-picked networks to run quick benchmarking with network_index = [447, 133, 122, 80].


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



## Usage

To run a benchmark:

1. Ensure you have the necessary dataset in the `data/` directory.
2. Use the `src/benchmark/bench_noise_filtering.py` script to benchmark a specific filter:

   ```python
   from src.benchmark.bench_noise_filtering import bench_noise_filtering
   from src.net_filtering.filter import Filter
   import networkx as nx

   # Load your network
   G = nx.read_edgelist("data/your_network.edgelist")

   # Create a filter instance
   filter_instance = Filter()

   # Run the benchmark
   jaccard_score = bench_noise_filtering(G, filter_instance.mst)
   print(f"Jaccard Score: {jaccard_score}")
   ```

## Benchmarking Process

1. The framework adds noise to the input network using the `add_noise_to_network` function.
2. It then applies the specified filtering technique to the noisy network.
3. The performance is evaluated by comparing the filtered network to the original network using metrics such as Jaccard similarity.

## Adding New Filters

To add a new filtering technique:

1. Implement your filter in the `src/net_filtering/filter.py` file.
2. Ensure your filter function takes a networkx Graph as input and returns a filtered Graph.
3. Use the `bench_noise_filtering` function to benchmark your new filter.






