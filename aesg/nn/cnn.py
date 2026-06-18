import torch
import torch.nn as nn
from aesg.memory.controller import AESGMemory

class AESG_CNNLayer(nn.Module):
    """
    Inyección de contexto AESG en un bottleneck convolucional o densa.
    """
    def __init__(self, in_channels: int, out_channels: int, memory: AESGMemory, map_size: int):
        super().__init__()
        self.memory = memory
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Proyecta los canales espaciales hacia el espacio conceptual
        self.query_proj = nn.Linear(in_channels * map_size * map_size, memory.vector_dim)
        
        # Proyecta el concepto recuperado hacia canales espaciales
        self.mem_inject = nn.Linear(memory.vector_dim, out_channels * map_size * map_size)
        self.map_size = map_size

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        
        # Generar query a partir del volumen espacial (simplificado)
        x_flat = x.view(B, -1)
        query_vector = self.query_proj(x_flat[0])
        
        # Recuperar contexto y colapsarlo (adaptador para CNN)
        context = self.memory.retrieve(query_vector)
        mem_vector = context.aggregate()
        
        # Inyectar
        mem_spatial = self.mem_inject(mem_vector).view(1, -1, self.map_size, self.map_size)
        mem_spatial = mem_spatial.expand(B, -1, -1, -1)
        
        # Convolución sobre la suma (o concatenación)
        out = self.conv(x) + mem_spatial
        return torch.relu(out)
