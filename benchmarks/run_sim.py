import os
import torch
import torch.nn as nn
from torch.optim import Adam
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aesg.memory.controller import AESGMemory
from aesg.config import AESGConfig
from aesg.nn.rnn import AESG_GRU
from aesg.trainer.dual_trainer import DualTrainer
from aesg.tools.explain import Explainer
from aesg.tools.evolution_tracker import EvolutionTracker

class ToyModel(nn.Module):
    def __init__(self, mem):
        super().__init__()
        self.gru = AESG_GRU(16, 32, mem)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        h = torch.zeros(x.size(0), 32)
        h = self.gru(x, h)
        return self.fc(h)

def run_simulation():
    # Limpiar directorio si existe
    if os.path.exists("./test_mem"):
        shutil.rmtree("./test_mem")
        
    print("Inicializando Memoria AESG V2...")
    config = AESGConfig(vector_dim=32, spreading_activation_steps=2)
    mem = AESGMemory("./test_mem", config=config)
    
    print("Creando modelo Toy con GRU...")
    model = ToyModel(mem)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    trainer = DualTrainer(model, optimizer, criterion)
    
    print("Simulando entrenamiento (10 iteraciones)...")
    for i in range(10):
        # Datos aleatorios simulados
        x = torch.randn(8, 16) 
        y = torch.randn(8, 1)
        
        loss = trainer.fit_step(x, y)
        if i % 2 == 0:
            print(f"Step {i} | Loss: {loss:.4f} | Nodos en grafo: {mem.storage.node_count}")
    
    print("\nEntrenamiento finalizado. Analizando interpretabilidad...")
    explainer = Explainer(mem)
    hubs = explainer.find_hubs(top_n=3)
    print("Top Hubs:", hubs)
    
    if hubs:
        info = explainer.explain(hubs[0])
        print("Información del Hub principal:")
        print(f"- Frecuencia de uso: {info.get('Use_Frequency')}")
        print(f"- N° Conexiones extraídas: {len(info.get('Main_Connections', []))}")
        
    print("\nRevisando Historial Evolutivo...")
    tracker = EvolutionTracker(mem.logger)
    print("Resumen de Eventos:", tracker.summary())
    tracker.print_abstract_creations()
        
    print("\n¡Simulación completada con éxito!")

if __name__ == "__main__":
    run_simulation()
