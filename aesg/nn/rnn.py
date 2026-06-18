import torch
import torch.nn as nn
from aesg.memory.controller import AESGMemory

class AESG_GRU(nn.Module):
    """
    Integración de AESG con una celda GRU.
    Utiliza un tensor colapsado (adaptador opcional) para alimentar la memoria al RNN.
    """
    def __init__(self, input_size: int, hidden_size: int, memory: AESGMemory):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory = memory
        
        # El input de la GRU es la concatenación de la entrada real + contexto de memoria
        self.gru_cell = nn.GRUCell(input_size + memory.vector_dim, hidden_size)
        
        # Generador de query a memoria desde el hidden state
        self.query_proj = nn.Linear(hidden_size, memory.vector_dim)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor):
        # 1. Generar query para la memoria basado en el estado anterior
        query_vector = self.query_proj(h_prev)
        
        # 2. Recuperar memoria
        # Se obtiene el RetrievedContext completo y se colapsa (adaptador para GRU)
        context = self.memory.retrieve(query_vector[0]) 
        memory_vector = context.aggregate()
        
        # Si el input viene por lotes, replicamos temporalmente (simplificado)
        if x.dim() > 1 and memory_vector.dim() == 1:
            memory_vector = memory_vector.unsqueeze(0).expand(x.size(0), -1)
            
        # 3. Concatenar entrada con memoria
        x_mem = torch.cat([x, memory_vector], dim=-1)
        
        # 4. Paso GRU
        h_new = self.gru_cell(x_mem, h_prev)
        
        return h_new


class AESG_LSTM(nn.Module):
    """
    Integración de AESG con una celda LSTM.
    Similar al GRU pero con estado de celda.
    """
    def __init__(self, input_size: int, hidden_size: int, memory: AESGMemory):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory = memory
        
        self.lstm_cell = nn.LSTMCell(input_size + memory.vector_dim, hidden_size)
        self.query_proj = nn.Linear(hidden_size, memory.vector_dim)

    def forward(self, x: torch.Tensor, hx: tuple):
        h_prev, c_prev = hx
        
        query_vector = self.query_proj(h_prev)
        context = self.memory.retrieve(query_vector[0])
        memory_vector = context.aggregate()
        
        if x.dim() > 1 and memory_vector.dim() == 1:
            memory_vector = memory_vector.unsqueeze(0).expand(x.size(0), -1)
            
        x_mem = torch.cat([x, memory_vector], dim=-1)
        
        h_new, c_new = self.lstm_cell(x_mem, (h_prev, c_prev))
        
        return (h_new, c_new)
