import torch
import torch.nn as nn
from torch.optim import Optimizer

class DualTrainer:
    """
    Entrenador dual para arquitecturas basadas en AESG.
    Actualiza simultáneamente los pesos neuronales (Backprop) 
    y la topología de la memoria (Hebbian/RL).
    """
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Encuentra el módulo AESGMemory dentro del modelo
        self.memory_module = None
        for module in self.model.modules():
            # Avoid direct import to avoid circular dependencies in complex setups, 
            # or just use class name matching.
            if type(module).__name__ == "AESGMemory":
                self.memory_module = module
                break
                
        if not self.memory_module:
            print("Warning: No se encontró AESGMemory en el modelo. El entrenamiento de topología será ignorado.")

    def fit_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        
        # V2: Resetear el estado activo de la memoria para la nueva secuencia/batch
        if self.memory_module:
            self.memory_module.reset_state()
        
        # 1. Forward Pass (Inferencia + Navegación de Memoria por Spreading Activation)
        outputs = self.model(inputs)
        
        # 2. Loss computation
        loss = self.criterion(outputs, targets)
        
        # 3. Backward Pass (Actualización de pesos neuronales)
        loss.backward()
        self.optimizer.step()
        
        # 4. Topology Update (Actualización de grafo AESG)
        if self.memory_module:
            self.memory_module.update_topology()
            
        return loss.item()
        
    def fit(self, dataloader, epochs: int):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                loss = self.fit_step(inputs, targets)
                epoch_loss += loss
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
            # Persistencia periódica de la memoria
            if self.memory_module:
                self.memory_module.save()
