
##  Benchmarking Network Filtering

This repository implements a framework for benchmarking various network filtering techniques. It also provides real and simulated networks and it implements common network filtering techniques for benchmarking purposes.

## Project Structure

- **src/:** Contains the source code for the benchmarking framework
  - **benchmark/:** Benchmarking utilities and core functionality
  - **net_filtering/:** Implementation of various network filtering techniques
- **data/:** Includes sample datasets and network traces for testing
  - **real_nets/:** Networks extracted from ICON
  - **simulated_nets/:** Simulated network datasets
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






