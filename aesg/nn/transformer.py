import torch
import torch.nn as nn
from aesg.memory.controller import AESGMemory

class AESG_TransformerLayer(nn.Module):
    """
    Capa de Transformer que interactúa con la estructura granular de la memoria.
    Evita colapsar los tensores; utiliza atención cruzada.
    """
    def __init__(self, d_model: int, nhead: int, memory: AESGMemory):
        super().__init__()
        self.memory = memory
        
        # Self attention for the sequence
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Cross attention to AESG memory vectors
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.query_proj = nn.Linear(d_model, memory.vector_dim)
        
        # Si d_model != memory.vector_dim, proyectamos
        self.mem_proj = nn.Linear(memory.vector_dim, d_model) if d_model != memory.vector_dim else nn.Identity()

    def forward(self, src: torch.Tensor):
        # 1. Self Attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.norm1(src2)
        
        # 2. Fetch memory context based on current states
        # Toma el promedio del batch/seq para generar un único query (o múltiples queries según la estrategia)
        # Aquí simplificamos tomando la media de la secuencia para el query
        seq_mean = src.mean(dim=1)
        query_vector = self.query_proj(seq_mean[0]) # batch_size=0 for simplicity
        
        # Devuelve RetrievedContext con todos los conceptos individuales
        context = self.memory.retrieve(query_vector)
        mem_vectors = context.vectors.unsqueeze(0) # (1, num_concepts, vec_dim)
        
        # Alinear dimensiones
        mem_keys = self.mem_proj(mem_vectors).expand(src.size(0), -1, -1) # (B, num_concepts, d_model)
        
        # 3. Cross Attention con múltiples conceptos
        src2, _ = self.cross_attn(query=src, key=mem_keys, value=mem_keys)
        src = src + self.norm2(src2)
        
        # 4. Feed Forward
        src2 = self.ff(src)
        src = src + self.norm3(src2)
        
        return src
