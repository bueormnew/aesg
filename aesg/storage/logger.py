import os
import time
import struct
from typing import List, Dict, Any

class EvolutionLogger:
    """
    Registra el historial evolutivo de la memoria AESG en formato binario (.aesglog).
    Diseñado para velocidad, persistencia y escalabilidad.
    """
    EVENT_TYPES = {
        "CREATE": 0,
        "MERGE": 1,
        "SPLIT": 2,
        "PRUNE": 3,
        "CONSOLIDATE": 4,
        "RESTRUCTURE": 5
    }
    INV_EVENT_TYPES = {v: k for k, v in EVENT_TYPES.items()}

    # Struct format: < Q B 3x Q Q f (32 bytes)
    # timestamp (8), type (1), padding (3), id1 (8), id2 (8), val (4)
    STRUCT_FORMAT = '< Q B 3x Q Q f'
    RECORD_SIZE = struct.calcsize(STRUCT_FORMAT)

    def __init__(self, directory: str):
        self.log_path = os.path.join(directory, "evolution.aesglog")
        
    def log_event(self, event_type: str, id1: int = 0, id2: int = 0, val: float = 0.0):
        """
        Guarda un evento evolutivo en el log binario.
        """
        t_stamp = int(time.time())
        type_code = self.EVENT_TYPES.get(event_type, 255)
        
        record = struct.pack(self.STRUCT_FORMAT, t_stamp, type_code, int(id1), int(id2), float(val))
        with open(self.log_path, 'ab') as f:
            f.write(record)
            
    def get_history(self) -> List[Dict[str, Any]]:
        history = []
        if not os.path.exists(self.log_path):
            return history
            
        with open(self.log_path, 'rb') as f:
            while True:
                data = f.read(self.RECORD_SIZE)
                if not data or len(data) < self.RECORD_SIZE:
                    break
                
                t_stamp, type_code, id1, id2, val = struct.unpack(self.STRUCT_FORMAT, data)
                
                history.append({
                    "timestamp": t_stamp,
                    "type": self.INV_EVENT_TYPES.get(type_code, "UNKNOWN"),
                    "id1": id1,
                    "id2": id2,
                    "val": val
                })
        return history
