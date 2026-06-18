import os
import sys
import time
import shutil
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory

def measure_latency_and_scale():
    print("--- INICIANDO BENCHMARK DE AESG V2.1 ---")
    
    # 1. Configuración adaptada a recursos limitados (Test 50,000 Nodos)
    TEST_DIR = "./benchmark_mem"
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        
    config = AESGConfig(
        vector_dim=128, 
        max_concepts=100_000, # Max capacity allocation
        spreading_activation_steps=2,
        novelty_birth_threshold=1
    )
    
    print("\n[1] Inicializando Storage (mmap)...")
    start_t = time.time()
    memory = AESGMemory(TEST_DIR, config=config)
    init_time = time.time() - start_t
    print(f"-> Inicialización completada en {init_time:.4f}s")
    
    # 2. Inserción Masiva
    print("\n[2] Prueba de Estrés: Creando 5,000 nodos (Concept Birth)...")
    start_t = time.time()
    
    # Simular inputs
    for i in range(5000):
        dummy_vector = torch.randn(128)
        # Fuerza nacimiento sensorial
        memory.cognitive_engine.create_sensory_concept(dummy_vector)
    
    insert_time = time.time() - start_t
    print(f"-> 5,000 conceptos insertados en {insert_time:.4f}s ({(5000/insert_time):.2f} ops/s)")
    
    # 3. Prueba de Conexiones
    print("\n[3] Prueba de Red: Creando 20,000 aristas...")
    start_t = time.time()
    for i in range(20000):
        source = np.random.randint(0, 5000)
        target = np.random.randint(0, 5000)
        target_id = memory.storage.nodes['id'][target]
        memory.storage.add_edge(source, target_id, weight=0.5)
    edge_time = time.time() - start_t
    print(f"-> 20,000 aristas creadas en {edge_time:.4f}s")
    
    memory.save()
    
    # Imprimir tamaño en disco
    nodes_size = os.path.getsize(os.path.join(TEST_DIR, "nodes.aesg")) / (1024*1024)
    edges_size = os.path.getsize(os.path.join(TEST_DIR, "edges.aesg")) / (1024*1024)
    log_size = os.path.getsize(os.path.join(TEST_DIR, "evolution.aesglog")) / (1024)
    print(f"\n[4] Tamaño en Disco:")
    print(f"-> nodes.aesg: {nodes_size:.2f} MB")
    print(f"-> edges.aesg: {edges_size:.2f} MB")
    print(f"-> evolution.aesglog: {log_size:.2f} KB")

    # 4. Latencia de Recuperación (Spreading Activation)
    print("\n[5] Benchmark de Navegación: 100 consultas consecutivas...")
    latencies = []
    
    # Preparar el contexto activo
    memory.reset_state()
    
    for _ in range(100):
        query = torch.randn(128)
        st = time.time()
        # Esto disparará Spreading Activation
        context = memory.retrieve(query)
        en = time.time()
        latencies.append((en - st) * 1000) # a milisegundos
        
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    print(f"-> Latencia Media por inferencia: {avg_latency:.2f} ms")
    print(f"-> Latencia p99: {p99_latency:.2f} ms")
    
    # 5. Limpieza para no estallar la PC
    print("\n[6] Limpieza de espacio en disco...")
    del memory
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    print("-> Directorio de prueba eliminado.")
    
    print("\n--- BENCHMARK FINALIZADO ---")

if __name__ == "__main__":
    measure_latency_and_scale()
