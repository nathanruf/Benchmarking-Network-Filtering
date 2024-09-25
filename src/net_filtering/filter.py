import networkx as nx

class Filter:
    def mst(self, graph):
        filteredGraph = nx.Graph()
        id = {node: node for node in graph.nodes}
        sz = {node: 1 for node in graph.nodes}

        sortedEdges = sorted(graph.edges(data = True), key = lambda x : x[2]['weight'])

        for edge in sortedEdges:
            if(self.__find(edge[0], id) == self.__find(edge[1], id)):
                continue
            self.__union(edge[0], edge[1], sz, id)
            filteredGraph.add_edge(edge[0], edge[1], weight = edge[2]['weight'])

        return filteredGraph

    def pmfg(self, graph):
        filteredGraph = nx.Graph()
        edgeLimit = 3 * (len(graph.nodes) - 2)
        sortedEdges = sorted(graph.edges(data = True), key = lambda i : i[2]['weight'])

        for edge in sortedEdges:
            filteredGraph.add_edge(edge[0], edge[1], weight = edge[2]['weight'])
            if not nx.check_planarity(filteredGraph):
                filteredGraph.remove_edge(edge[0], edge[1])
            
            if len(filteredGraph.edges) == edgeLimit:
                break

        return filteredGraph

    def threshold(self, graph, threshold):
        filteredGraph = nx.Graph()

        for u, v, edge in graph.edges(data = True):
            if edge.get('weight', 0) >= threshold:
                filteredGraph.add_edge(u, v, weight = edge['weight'])

        return filteredGraph
    
    #Métodos auxiliares para o algoritmo MST
    def __find(self, p, id):
        if id[p] == p:
            return p
        id[p] = self.__find(id[p], id)
        return id[p]
    
    def __union(self, p, q, sz, id):
        p = self.__find(p, id)
        q = self.__find(q, id)

        if p == q:
            return

        #swap(p,q)
        if(sz[p] > sz[q]):
            p, q = q, p

        id[p] = q
        sz[q] += sz[p]

        return