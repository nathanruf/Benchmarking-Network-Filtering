## Results classes:
classe 1 - gráfico para cada benchmark - eixo x - noise level e eixo y jaccard para cada filtro - com redes de tamanho 1000 - 4 plots pra cada benchmark

classe 2 - mesma coisa para precision, recall e f1_score

classe 3 - assortativity, clustering, average degree, density - cosine distance

classe 4 - variação das métricas acima pelo percentual - fazer boxplot de todos os tamanhos

classe 5 - colocar o valor original no título e o gráfico com o filtrado de todas as métricas

##Future todo
x10 size network

## Comparing Barabási-Albert (BA) and Watts-Strogatz (WS) networks


### Degree-Related Metrics:

BA Networks: High variance in degree distribution, power-law scaling
WS Networks: Low variance, more uniform degree distribution
Measurable through: degree distribution, variance, maximum degree


### Clustering Properties:

BA Networks: Low clustering (0.02-0.15)
WS Networks: High clustering (0.3-0.6)
Measurable through: clustering coefficient, local clustering


### Path Lengths:

BA Networks: Shorter average paths (3-4 steps)
WS Networks: Slightly longer paths (4-6 steps)
Measurable through: average path length, diameter


### Centrality Measures:

BA Networks: Higher betweenness centrality variance (hub effect)
WS Networks: More uniform betweenness distribution
Measurable through: betweenness centrality, closeness centrality


### Network Efficiency:

BA Networks: Higher global efficiency (0.25-0.35)
WS Networks: Lower global efficiency (0.15-0.25)
Measurable through: global efficiency metric


### Structural Properties:

BA Networks: Negative or neutral assortativity
WS Networks: Slightly positive assortativity
Measurable through: degree assortativity coefficient

### Sample Code

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def compare_network_metrics(n_nodes=1000, m_ba=3, k_ws=6, p_ws=0.1):
    """
    Generate and compare metrics for BA and WS networks
    
    Parameters:
    n_nodes: Number of nodes
    m_ba: Number of edges to attach from new node to existing nodes (BA)
    k_ws: Each node is connected to k nearest neighbors (WS)
    p_ws: Probability of rewiring each edge (WS)
    """
    # Generate networks
    ba_network = nx.barabasi_albert_graph(n_nodes, m_ba)
    ws_network = nx.watts_strogatz_graph(n_nodes, k_ws, p_ws)
    
    metrics = {
        'BA Network': {},
        'WS Network': {}
    }
    
    # Calculate metrics for both networks
    for name, G in [('BA Network', ba_network), ('WS Network', ws_network)]:
        # Basic metrics
        metrics[name]['Average Degree'] = np.mean([d for n, d in G.degree()])
        metrics[name]['Average Clustering'] = nx.average_clustering(G)
        metrics[name]['Average Path Length'] = nx.average_shortest_path_length(G)
        metrics[name]['Diameter'] = nx.diameter(G)
        
        # Degree distribution
        degrees = [d for n, d in G.degree()]
        metrics[name]['Degree Variance'] = np.var(degrees)
        metrics[name]['Maximum Degree'] = max(degrees)
        
        # Centrality measures
        metrics[name]['Average Betweenness'] = np.mean(list(nx.betweenness_centrality(G).values()))
        metrics[name]['Average Closeness'] = np.mean(list(nx.closeness_centrality(G).values()))
        
        # Network efficiency
        metrics[name]['Global Efficiency'] = nx.global_efficiency(G)
        
        # Assortativity
        metrics[name]['Degree Assortativity'] = nx.degree_assortativity_coefficient(G)
    
    return metrics

# Example usage and typical values
metrics = compare_network_metrics()

# Typical ranges for different network sizes
"""
Metric Ranges (n=1000 nodes):

BA Networks:
- Average Degree: 4-8
- Clustering Coefficient: 0.02-0.15
- Average Path Length: 3-4
- Diameter: 6-8
- Degree Variance: High (100-1000)
- Maximum Degree: High (50-100)
- Betweenness Centrality: 0.02-0.05
- Global Efficiency: 0.25-0.35
- Degree Assortativity: -0.1 to 0.1

WS Networks:
- Average Degree: 4-8
- Clustering Coefficient: 0.3-0.6
- Average Path Length: 4-6
- Diameter: 8-12
- Degree Variance: Low (1-10)
- Maximum Degree: Low (8-15)
- Betweenness Centrality: 0.01-0.03
- Global Efficiency: 0.15-0.25
- Degree Assortativity: 0.0 to 0.3
"""
```
