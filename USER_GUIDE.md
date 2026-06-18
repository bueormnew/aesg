# 📖 AESG - Guía Rápida del Usuario

AESG (Adaptive External Semantic Graph) es una librería que te permite dotar a tus redes neuronales de una **memoria a largo plazo** que evoluciona, aprende y olvida por sí sola.

Esta guía está diseñada para que integres AESG en tus proyectos en 5 minutos, sin preocuparte por la teoría profunda.

## 1. Instalación
```bash
pip install aesg
```

## 2. Inicializar la Memoria
Toda la memoria se guarda automáticamente en disco en una carpeta que tú elijas.

```python
from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

# Configuramos la memoria para aceptar vectores de tamaño 256
config = AESGConfig(vector_dim=256)

# Inicializamos (creará la carpeta si no existe)
memoria = AESGMemory(directory="./mi_cerebro", config=config)
```

## 3. Consultar la Memoria en Inferencia
Simplemente pásale un tensor de PyTorch (tu input, una imagen procesada, texto, etc.).

```python
import torch

vector_entrada = torch.randn(256)

# Buscar en la memoria
contexto = memoria(vector_entrada)

print("IDs encontrados:", contexto.primary_concepts)
print("Vectores recuperados:", contexto.concept_vectors.shape)
```
> **Nota:** Si la memoria no conoce ese vector (le "sorprende"), lo aprenderá automáticamente.

## 4. Entrenar tu Modelo con AESG
Para que tu red neuronal aprenda a usar la memoria, debes usar nuestro `DualTrainer`. Este entrenador no solo actualiza los pesos de tu red neuronal, sino que también ordena a la memoria que haga "limpieza" (fusione conceptos repetidos y olvide los que no sirven).

```python
from aesg.trainer.dual_trainer import DualTrainer
from torch.optim import Adam
import torch.nn as nn

# Tu modelo de PyTorch
mi_red = MiModelo(memoria) 
optimizer = Adam(mi_red.parameters(), lr=0.001)

# Usar el entrenador de AESG
entrenador = DualTrainer(model=mi_red, optimizer=optimizer, criterion=nn.MSELoss())

# Entrenar un paso (Forward, Backward, Step)
entrenador.fit_step(inputs=datos, targets=etiquetas)

# Al final de una época, dile a la memoria que se organice
memoria.update_topology()
```

## 5. Visualizar qué aprendió la memoria
AESG guarda un registro histórico (`.aesglog`) de todo lo que aprende y olvida.

```python
from aesg.tools.evolution_tracker import EvolutionTracker

tracker = EvolutionTracker(memoria.logger)
print("Resumen de vida de la memoria:", tracker.summary())
tracker.plot_event_distribution() # Muestra una gráfica
```

¡Eso es todo! Con esto puedes integrar memoria persistente en cualquier agente o modelo.
