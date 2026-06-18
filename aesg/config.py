from dataclasses import dataclass

@dataclass
class AESGConfig:
    """Configuración global para AESG V2."""
    
    vector_dim: int = 256
    
    # Presión Evolutiva (Presupuestos y Límites)
    max_concepts: int = 1_000_000
    max_edges_per_node: int = 1000
    survival_threshold_relevance: float = 0.05
    survival_threshold_frequency: int = 5
    
    # Umbrales Cognitivos
    coactivation_threshold_create: int = 50 # Cuántas coactivaciones para crear nodo abstracto
    merge_similarity_threshold: float = 0.95 # Similitud de vecindad para fusionar
    split_density_threshold: int = 800 # Aristas máximas antes de considerar partición
    
    # Curiosidad Adaptativa y Novedad
    novelty_explanation_threshold: float = 0.6 # Puntuación mínima de explicación del grafo (0.0-1.0)
    novelty_birth_threshold: int = 3 # Cuántas veces debe persistir la sorpresa para crear el nodo
    
    # Navegación (Spreading Activation)
    spreading_activation_decay: float = 0.8
    spreading_activation_steps: int = 3
    region_facilitation_multiplier: float = 1.5 # Multiplicador para navegación intra-región
    
    # Consolidación
    consolidation_epoch_interval: int = 10
    prune_confidence_threshold: float = 0.1
