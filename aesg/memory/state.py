from typing import Set, List

class BatchActiveContext:
    """
    Mantiene el estado independiente de la memoria para cada muestra de un batch.
    Representa los "nodos activos actualmente" en la navegación conceptual.
    """
    def __init__(self, batch_size: int):
        # Lista de conjuntos (uno por elemento del batch)
        # Cada conjunto contiene los índices de los nodos actualmente activos.
        self.active_nodes: List[Set[int]] = [set() for _ in range(batch_size)]
        
        # Buffer de novedad: Cuenta veces consecutivas que una sorpresa es detectada
        self.novelty_buffer: List[int] = [0 for _ in range(batch_size)]
        
    def get_active(self, batch_idx: int) -> Set[int]:
        return self.active_nodes[batch_idx]
        
    def set_active(self, batch_idx: int, nodes: Set[int]):
        self.active_nodes[batch_idx] = nodes

    def add_active(self, batch_idx: int, nodes: Set[int]):
        self.active_nodes[batch_idx].update(nodes)
        
    def clear(self):
        for s in self.active_nodes:
            s.clear()
        for i in range(len(self.novelty_buffer)):
            self.novelty_buffer[i] = 0
