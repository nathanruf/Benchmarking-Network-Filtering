import networkx as nx
import numpy as np
import random
import scipy
import heapdict
from itertools import combinations
from sklearn.cluster import KMeans

"""
This class implements network filtering and graph sparsification techniques.

Methods:
- mst: Computes the Minimum Spanning Tree of a graph
- pmfg: Computes the Planar Maximally Filtered Graph
- threshold: Applies a global threshold filter to the graph
- local_degree_sparsifier: Sparsifies graph based on local node degrees
- random_edge_sparsifier: Randomly sparsifies edges
- simmelian_sparsifier: Implements Simmelian backbone sparsification
- disparity_filter: Implements the disparity filter technique
- overlapping_trees: Implements the Overlapping Trees network reduction technique
- k_core_decomposition: Implements k-core decomposition network reduction

Dependencies:
- networkx
- numpy
- scipy
"""
class Filter:
    def mst(self, graph: nx.Graph) -> nx.Graph:
        """
        Computes the Minimum Spanning Tree of a graph.

        Args:
            graph (nx.Graph): Input graph

        Returns:
            nx.Graph: Minimum Spanning Tree
        """

        def find(p: int, id: dict) -> int:
            """
            Helper method for the MST algorithm (Union-Find).

            Args:
                p (int): Node index
                id (dict): Dictionary of node identifiers

            Returns:
                int: Root node identifier
            """
            if id[p] == p:
                return p
            id[p] = find(id[p], id)
            return id[p]
        
        def union(p: int, q: int, sz: dict, id: dict) -> None:
            """
            Helper method for the MST algorithm (Union-Find).

            Args:
                p (int): First node index
                q (int): Second node index
                sz (dict): Dictionary of set sizes
                id (dict): Dictionary of node identifiers
            """
            p = find(p, id)
            q = find(q, id)

            if p == q:
                return

            if sz[p] > sz[q]:
                p, q = q, p

            id[p] = q
            sz[q] += sz[p]

        filteredGraph = nx.Graph()
        id = {node: node for node in graph.nodes}
        sz = {node: 1 for node in graph.nodes}

        sortedEdges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])

        for edge in sortedEdges:
            if find(edge[0], id) == find(edge[1], id):
                continue
            union(edge[0], edge[1], sz, id)
            filteredGraph.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        return filteredGraph

    def pmfg(self, graph: nx.Graph) -> nx.Graph:
        """
        Computes the Planar Maximally Filtered Graph.

        Args:
            graph (nx.Graph): Input graph

        Returns:
            nx.Graph: Planar Maximally Filtered Graph
        """
        filteredGraph = nx.Graph()
        filteredGraph.add_nodes_from(graph.nodes)
        edgeLimit = 3 * (len(graph.nodes) - 2)
        sortedEdges = sorted(graph.edges(data=True), key=lambda i: i[2]['weight'])

        for edge in sortedEdges:
            filteredGraph.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
            if not nx.check_planarity(filteredGraph):
                filteredGraph.remove_edge(edge[0], edge[1])
            
            if len(filteredGraph.edges) == edgeLimit:
                break

        return filteredGraph
    
    def tmfg(self, original_graph:nx.Graph) -> nx.Graph:
        def initial_tetrahedron(edge_weights: dict) -> tuple:
            max_score = -1
            best_tetrahedron = None
            for tetrahedron in combinations(original_graph.nodes(), 4):
                score = 0
                for edge in combinations(tetrahedron, 2):
                    score += edge_weights.get(frozenset(edge), 0)
                if score > max_score:
                    max_score = score
                    best_tetrahedron = tetrahedron
            return best_tetrahedron

        def update_best_tetrahedron(best_tetrahedron:heapdict, edge_weights:dict, v:list, original_face:list, new_vertex:int) -> None:
            updated_face = list(original_face)
            updated_face.append(new_vertex)
            tetrahedron = set(updated_face)

            for face in combinations(tetrahedron, 3):
                if new_vertex not in face:
                    continue
                for vertex in v:
                    tetrahedron = list(face)
                    tetrahedron.insert(0, vertex)

                    score = 0
                    for edge in combinations(tetrahedron, 2):
                        score += edge_weights.get(frozenset(edge), 0)

                    best_tetrahedron[tuple(tetrahedron)] = -score

        def get_best_tetrahedron(best_tetrahedron:heapdict) -> tuple[list, int]:
            tetrahedron, _ = best_tetrahedron.peekitem()
            new_vertex, *face = tetrahedron
            
            combinations_to_remove = [key for key in best_tetrahedron.keys() if new_vertex in key or all(v in key for v in face)]
            for combination in combinations_to_remove:
                del best_tetrahedron[combination]

            return face, new_vertex

        def update_filtered_graph(edge_weights:dict, filtered_graph:nx.Graph, face:list, new_vertex:int) -> None:
            for node in face:
                edge = (node, new_vertex)
                filtered_graph.add_edge(node, new_vertex, weight = edge_weights.get(frozenset(edge), 0))

        def preprocess_edge_weights() -> dict:
            edge_weights = {frozenset((u,v)): data['weight'] for u, v, data in original_graph.edges(data=True)}
            return edge_weights

        edge_weights = preprocess_edge_weights()

        c1 = initial_tetrahedron()
        
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(original_graph.nodes())

        v = [node for node in original_graph.nodes() if node not in c1]
        best_tetrahedron = heapdict.heapdict()

        for face in combinations(c1, 3):
            for edge in combinations(face, 2):
                data = edge_weights.get(frozenset(edge), 0)
                filtered_graph.add_edge(edge[0], edge[1], weight = data)
            update_best_tetrahedron(best_tetrahedron, edge_weights, v, face, face[0])

        while len(v) != 0:
            face, new_vertex = get_best_tetrahedron(best_tetrahedron)
            v.remove(new_vertex)
            update_best_tetrahedron(best_tetrahedron, edge_weights, v, face, new_vertex)
            update_filtered_graph(edge_weights, filtered_graph, face, new_vertex)

        return filtered_graph

    def threshold(self, graph: nx.Graph, threshold: float) -> nx.Graph:
        """
        Applies a global threshold filter to the graph.

        Args:
            graph (nx.Graph): Input graph
            threshold (float): Threshold value for edge weights

        Returns:
            nx.Graph: Filtered graph
        """
        filteredGraph = nx.Graph()
        filteredGraph.add_nodes_from(graph.nodes())

        for u, v, edge in graph.edges(data=True):
            if edge.get('weight', 0) >= threshold:
                filteredGraph.add_edge(u, v, weight=edge['weight'])

        return filteredGraph
    
    def local_degree_sparsifier(self, G: nx.Graph, target_ratio: float) -> nx.Graph:
        """
        Sparsifies graph based on local node degrees.

        Args:
            G (nx.Graph): Input graph
            target_ratio (float): Target ratio of edges to keep

        Returns:
            nx.Graph: Sparsified graph
        """
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        
        edges = sorted(G.edges(data=True), 
                       key=lambda x: min(G.degree(x[0]), G.degree(x[1])),
                       reverse=True)
        
        target_edges = int(G.number_of_edges() * target_ratio)
        H.add_edges_from(edges[:target_edges])
        
        return H

    def random_edge_sparsifier(self, G: nx.Graph, target_ratio: float, seed: int = 42) -> nx.Graph:
        """
        Randomly sparsifies edges.

        Args:
            G (nx.Graph): Input graph
            target_ratio (float): Target ratio of edges to keep
            seed (int): Random seed for reproducibility

        Returns:
            nx.Graph: Sparsified graph
        """
        # Set seeds for both numpy and Python's random
        np.random.seed(seed)
        random.seed(seed)

        H = G.copy()
        
        num_to_remove = int(G.number_of_edges() * (1 - target_ratio))
        edges_to_remove = list(G.edges())
        random.shuffle(edges_to_remove)  # Use Python's random.shuffle
        edges_to_remove = edges_to_remove[:num_to_remove]
        
        H.remove_edges_from(edges_to_remove)
        return H

    def simmelian_sparsifier(self, G: nx.Graph, max_rank: int = 5) -> nx.Graph:
        """
        Implements Simmelian backbone sparsification.

        Args:
            G (nx.Graph): Input graph
            max_rank (int): Maximum rank considered for overlap calculation

        Returns:
            nx.Graph: Sparsified graph
        """
        def simmelian_strength(u, v):
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            common_neighbors = u_neighbors.intersection(v_neighbors)
            return len(common_neighbors)
        
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        
        for u in G.nodes():
            neighbors = sorted(G.neighbors(u), 
                               key=lambda x: simmelian_strength(u, x),
                               reverse=True)
            H.add_edges_from((u, v) for v in neighbors[:max_rank])
        
        return H

    def disparity_filter(self, G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
        """
        Implements the disparity filter technique as described in 
        Serrano et al. (2009) PNAS paper.

        Args:
            G (nx.Graph): Input weighted graph
            alpha (float): Significance level for the filter

        Returns:
            nx.Graph: Filtered graph
        """
        H = nx.Graph()
        H.add_nodes_from(G.nodes())

        for u in G.nodes():
            k = G.degree(u)
            if k > 1:
                strength = sum(G[u][v].get('weight', 1) for v in G[u])
                for v in G[u]:
                    weight = G[u][v].get('weight', 1)
                    p_ij = weight / strength
                    alpha_ij = 1 - (k - 1) * scipy.integrate.quad(lambda x: (1 - x)**(k-2), 0, p_ij)[0]
                    if alpha_ij < alpha:
                        H.add_edge(u, v, weight=weight)

        return H

    def overlapping_trees(self, G: nx.Graph, num_trees: int = 3) -> nx.Graph:
        """
        Implements the Overlapping Trees network reduction technique as described in
        Carmi et al. (2008) arXiv:0812.3227.

        This method creates a reduced network by combining multiple spanning trees.

        Args:
            G (nx.Graph): Input weighted graph
            num_trees (int): Number of spanning trees to generate and combine

        Returns:
            nx.Graph: Reduced graph
        """
        H = nx.Graph()
        H.add_nodes_from(G.nodes())

        for _ in range(num_trees):
            for (u, v, d) in G.edges(data=True):
                d['random_weight'] = np.random.random()

            T = nx.minimum_spanning_tree(G, weight='random_weight')
            H.add_edges_from(T.edges(data=True))

        for (u, v, d) in H.edges(data=True):
            d['weight'] = G[u][v].get('weight', 1)

        return H

    def k_core_decomposition(self, G: nx.Graph, k: int = None) -> nx.Graph:
        """
        Implements the k-core decomposition network reduction technique.

        This method creates a reduced network by recursively removing nodes with degree less than k,
        until no such nodes remain. If k is not specified, it returns the main core (largest k-core).

        Args:
            G (nx.Graph): Input graph
            k (int, optional): The order of the core. If not specified, returns the main core.

        Returns:
            nx.Graph: Reduced graph (k-core subgraph)

        References:
            Batagelj, V., & Zaversnik, M. (2003). An O(m) Algorithm for Cores Decomposition of Networks.
            https://arxiv.org/abs/cs.DS/0310049
        """
        core_numbers = nx.core_number(G)

        if k is None:
            k = max(core_numbers.values())

        H = G.subgraph([n for n, cn in core_numbers.items() if cn >= k])
        
        filtered_graph = nx.Graph()

        filtered_graph.add_nodes_from(G)
        filtered_graph.add_edges_from(H.edges(data=True))

        return filtered_graph
    
    def coarse_graining_espectral(self, G: nx.Graph) -> nx.Graph:
        # Compute the Laplacian matrix
        laplacian = nx.laplacian_matrix(G).toarray()

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Identify the spectral gap
        gaps = np.diff(eigenvalues)
        biggest_gap = np.argmax(gaps)  # Largest spectral gap

        # Define the number of groups based on the spectral gap
        n_groups = biggest_gap + 1
        embedding = eigenvectors[:, :n_groups]  # Embedding based on the first eigenvectors

        # KMeans clustering to group nodes
        kmeans = KMeans(n_clusters=n_groups, random_state=0)
        clusters = kmeans.fit_predict(embedding)

        # Group nodes into super-nodes
        super_nodes = {}
        for node, group in enumerate(clusters):
            if group not in super_nodes:
                super_nodes[group] = []
            super_nodes[group].append(node)

        # Create the renormalized graph
        G_coarse = nx.Graph()

        # Add super-nodes to the renormalized graph
        for group, nodes in super_nodes.items():
            G_coarse.add_node(group, original_nodes=nodes)

        # Add edges between super-nodes
        for u, v in G.edges():
            group_u = clusters[u]
            group_v = clusters[v]
            if group_u != group_v:
                if G_coarse.has_edge(group_u, group_v):
                    G_coarse[group_u][group_v]['weight'] += 1
                else:
                    G_coarse.add_edge(group_u, group_v, weight=1)

        # Rescale the adjacency matrix
        lambda_k = eigenvalues[biggest_gap]  # λk value
        adjacency_matrix = nx.adjacency_matrix(G_coarse).toarray()
        rescaled_matrix = (1 / lambda_k) * adjacency_matrix

        # Compute the renormalization error
        renormalization_error = np.linalg.norm(adjacency_matrix - rescaled_matrix)

        # Adjust weights based on the renormalization error
        for u, v, data in G_coarse.edges(data=True):
            data['weight'] = data['weight'] / (1 + renormalization_error)

        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(G.nodes())
        filtered_graph.add_edges_from(G_coarse.edges())

        return filtered_graph