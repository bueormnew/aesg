import os
import networkx as nx
from aesg.memory.controller import AESGMemory

class Exporter:
    def __init__(self, memory: AESGMemory):
        self.memory = memory
        self.graph = memory.graph
        
    def _to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        n_count = self.memory.storage.node_count
        
        for i in range(n_count):
            node = self.memory.storage.nodes[i]
            G.add_node(
                int(node['id']), 
                relevance=float(node['relevance']),
                use_frequency=int(node['use_frequency'])
            )
            
            edges = self.memory.storage.get_edges(i)
            for e in edges:
                G.add_edge(
                    int(e['source_id']), 
                    int(e['target_id']), 
                    weight=float(e['weight']),
                    confidence=float(e['confidence'])
                )
        return G

    def export_gml(self, filepath: str):
        G = self._to_networkx()
        nx.write_gml(G, filepath)
        
    def export_graphml(self, filepath: str):
        G = self._to_networkx()
        nx.write_graphml(G, filepath)
        
    def visualize(self):
        """
        Visualización local básica con NetworkX y Matplotlib (requiere matplotlib).
        Para grafos grandes se recomienda exportar a Gephi.
        """
        try:
            import matplotlib.pyplot as plt
            G = self._to_networkx()
            plt.figure(figsize=(10,10))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_size=20, alpha=0.5, with_labels=False)
            plt.show()
        except ImportError:
            print("Matplotlib no está instalado. Use export_gml y ábralo en Gephi.")
