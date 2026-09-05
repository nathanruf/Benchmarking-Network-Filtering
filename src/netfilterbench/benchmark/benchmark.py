"""
This module provides benchmarking tools for evaluating network filtering algorithms.

Functions:
- bench_noise_filtering: Benchmarks a network filter's ability to reduce noise.
- bench_structural_noise_filtering: Measures the ability of net_filter to reduce noise after an initial structural filtering.

Note:
    All functions expect undirected graphs (nx.Graph). Directed graphs (nx.DiGraph)
    should be converted before being passed to these functions.

Dependencies:
- networkx
- numpy
"""

import networkx as nx
import numpy as np


def _get_name(f: callable) -> str:
    """
    Get the name of a callable, supporting both regular functions and functools.partial objects.

    Args:
        f (callable): The callable to get the name of.

    Returns:
        str: The name of the callable.
    """
    return getattr(f, '__name__', None) or f.func.__name__


def __add_noise_to_network(input_net: nx.Graph, noise_level: float, seed: int = 42) -> nx.Graph:
    """
    Add noise to the network by adding random edges between unconnected vertices.

    Args:
        input_net (nx.Graph): The input network.
        noise_level (float): The ratio of random edges to total edges in the resulting graph.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        nx.Graph: The network with added noise.
    """
    np.random.seed(seed)
    noisy_net = input_net.copy()
    num_nodes = noisy_net.number_of_nodes()
    current_edges = noisy_net.number_of_edges()

    # Calculate the number of edges to add
    total_edges_after_noise = int(current_edges / (1 - noise_level))
    num_edges_to_add = total_edges_after_noise - current_edges

    # Ensure we don't exceed the maximum possible edges
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges_to_add = min(num_edges_to_add, max_possible_edges - current_edges)

    if num_edges_to_add <= 0:
        return noisy_net

    nodes_list = list(noisy_net.nodes())

    # Only assign a weight to injected noise edges if the original network was
    # itself weighted. An unweighted input should stay unweighted -- adding a
    # random weight attribute to noise edges (and not to the real ones) would
    # be meaningless and would make the graph inconsistently attributed.
    existing_weights = [d['weight'] for _, _, d in noisy_net.edges(data=True) if 'weight' in d]
    is_weighted = bool(existing_weights)
    min_weight, max_weight = (min(existing_weights), max(existing_weights)) if is_weighted else (0, 1)

    def _add_noise_edge(net, u, v):
        if is_weighted:
            net.add_edge(u, v, weight=np.random.uniform(min_weight, max_weight))
        else:
            net.add_edge(u, v)

    current_density = current_edges / max_possible_edges

    if current_density > 0.6:
        # For dense graphs: find non-edges explicitly to avoid infinite while loops
        non_edges = list(nx.non_edges(noisy_net))
        chosen_indices = np.random.choice(len(non_edges), size=num_edges_to_add, replace=False)
        for idx in chosen_indices:
            u, v = non_edges[idx]
            _add_noise_edge(noisy_net, u, v)
    else:
        # For sparse graphs: standard sampling (fast and memory-efficient)
        # Sample integer indices into nodes_list rather than the node labels
        # themselves -- np.random.choice requires 1-D input, which breaks for
        # non-scalar node labels (e.g. the (i, j) tuples used by grid graphs).
        n_nodes = len(nodes_list)
        edges_added = 0
        while edges_added < num_edges_to_add:
            i, j = np.random.choice(n_nodes, 2, replace=False)
            u, v = nodes_list[i], nodes_list[j]
            if not noisy_net.has_edge(u, v):
                _add_noise_edge(noisy_net, u, v)
                edges_added += 1

    return noisy_net


def bench_noise_filtering(input_net: nx.Graph, net_filter: callable, indicator_funcs: list[callable],
                           noise_level: float = 0.25, seed: int = 42) -> dict:
    """
    Measure ability of net_filter to reduce noise in input_net.

    Adds noise to the network, applies the filter function, and computes network-to-network
    indicators between the original and filtered networks.

    Args:
        input_net (nx.Graph): The input network.
        net_filter (callable): The network filter function.
        indicator_funcs (list[callable]): Indicator functions to be used.
        noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: The results of the indicator functions, keyed by function name.
    """
    noisy_net = __add_noise_to_network(input_net, noise_level, seed)
    filtered_net = net_filter(noisy_net)

    return {_get_name(f): f(input_net, filtered_net) for f in indicator_funcs}


def bench_structural_noise_filtering(input_net: nx.Graph, net_filter: callable, indicator_funcs: list[callable],
                                     noise_level: float = 0.25, seed: int = 42) -> dict:
    """
    Measure the ability of net_filter to reduce noise after an initial structural filtering.

    First applies the filter to the input network to obtain the structural network, then adds
    noise to it. Applies the filter again to reduce the noise, and computes network-to-network
    indicators between the structural and filtered networks.

    Args:
        input_net (nx.Graph): The input network.
        net_filter (callable): The network filter function.
        indicator_funcs (list[callable]): Indicator functions to be used.
        noise_level (float, optional): The ratio of random edges to total edges in the resulting graph. Defaults to 0.25.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: The results of the indicator functions, keyed by function name.
    """
    structural_net = net_filter(input_net)
    noisy_net = __add_noise_to_network(structural_net, noise_level, seed)
    filtered_net = net_filter(noisy_net)

    return {_get_name(f): f(structural_net, filtered_net) for f in indicator_funcs}