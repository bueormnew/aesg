import matplotlib.pyplot as plt
from collections import Counter
from aesg.storage.logger import EvolutionLogger

class EvolutionTracker:
    """
    Analiza y visualiza el historial evolutivo binario de AESG (.aesglog).
    """
    def __init__(self, logger: EvolutionLogger):
        self.logger = logger
        
    def plot_event_distribution(self):
        """Muestra la distribución de eventos evolutivos."""
        history = self.logger.get_history()
        events = [record["type"] for record in history]
        counter = Counter(events)
        
        plt.figure(figsize=(8, 5))
        plt.bar(counter.keys(), counter.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
        plt.title("Distribución de Eventos Evolutivos (AESG V2.1)")
        plt.ylabel("Frecuencia")
        plt.show()

    def print_abstract_creations(self):
        """Imprime el registro de cómo surgieron conceptos superiores (y conceptos sensoriales)."""
        history = self.logger.get_history()
        
        creations = [r for r in history if r["type"] == "CREATE"]
        abstract = [r for r in creations if r["val"] > 1.0]
        sensory = [r for r in creations if r["val"] == 1.0]
        
        print(f"--- Nacimientos Sensoriales (Novedad): {len(sensory)} ---")
        
        print(f"--- Creación de Conceptos Abstractos: {len(abstract)} ---")
        for c in abstract:
            # En la versión binaria, val guarda el número de padres
            # id1 es el nuevo nodo, id2 es el primer padre (por simplificación de bytes)
            print(f"Concepto {c['id1']} emergió combinando {int(c['val'])} nodos (incluyendo el {c['id2']})")
            
    def summary(self):
        history = self.logger.get_history()
        events = [record["type"] for record in history]
        counter = Counter(events)
        return dict(counter)
