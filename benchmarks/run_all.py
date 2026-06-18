"""
AESG V2.1 – Suite de Benchmarks Científicos
============================================
• B1: Escalabilidad de conceptos (inserción masiva)
• B2: Latencia de retrieve (Spreading Activation)
• B3: Estrés de Consolidación / Presión Evolutiva
• B4: Curiosidad Adaptativa (novelty filter)
• B5: Evolution Log (velocidad binaria)
• B6: Stress Test de RAM + Disco

Todos los benchmarks limpian sus archivos después de terminar.
El cleanup en Windows requiere del operador `del` + gc.collect() antes de rmtree.
"""

import os, sys, gc, shutil, time, json, struct
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from aesg.config import AESGConfig
from aesg.memory.controller import AESGMemory
from aesg.storage.logger import EvolutionLogger

# ── Directorios de salida ─────────────────────────────────────────────────────
RESULTS = os.path.join(ROOT, 'benchmarks', 'results')
IMAGES  = os.path.join(ROOT, 'docs', 'images')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(IMAGES,  exist_ok=True)

TMP_BASE = os.path.join(ROOT, '_bm_tmp')

def tmp_dir(name: str) -> str:
    d = os.path.join(TMP_BASE, name)
    os.makedirs(d, exist_ok=True)
    return d

def safe_del(mem):
    """Cierra mmap de forma segura antes de rmtree en Windows."""
    try:
        mem.storage.nodes._mmap.close()
        mem.storage.edges._mmap.close()
    except Exception:
        pass
    del mem
    gc.collect()

def clean(name: str):
    d = os.path.join(TMP_BASE, name)
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
        except Exception as e:
            print(f"  [warn] No se pudo limpiar {d}: {e}")

def ram_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

DB = {}   # Resultados acumulados

# ─────────────────────────────────────────────────────────────────────────────
# B1 – Escalabilidad de inserción de conceptos
# ─────────────────────────────────────────────────────────────────────────────
def b1_scalability():
    print("\n[B1] Escalabilidad de Conceptos")
    sizes   = [5_000, 50_000, 200_000, 500_000]
    results = []

    for n in sizes:
        dname = f'b1_{n}'
        clean(dname)
        cfg = AESGConfig(vector_dim=16, max_concepts=n + 500)
        mem = AESGMemory(tmp_dir(dname), config=cfg)

        vecs = torch.randn(n, 16)
        t0 = time.perf_counter()
        for i in range(n):
            mem.cognitive_engine.create_sensory_concept(vecs[i])
        elapsed = time.perf_counter() - t0
        ops = n / elapsed

        # Tamaño en disco
        mem.save()
        disk = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(tmp_dir(dname)) for f in fs
        ) / 1024**2

        r0 = ram_mb()
        results.append({'n': n, 'time_s': round(elapsed, 3),
                        'ops_s': round(ops, 1), 'disk_mb': round(disk, 2),
                        'ram_mb': round(r0, 1)})
        print(f"  {n:>7} nodos → {elapsed:.2f}s  ({ops:.0f} ops/s)  disco={disk:.1f}MB")

        safe_del(mem)
        clean(dname)

    DB['b1_scalability'] = results

    # Gráfica
    fig, ax = plt.subplots()
    ax.plot([r['n'] for r in results], [r['time_s'] for r in results],
            marker='o', color='steelblue', linewidth=2)
    ax.set_title('B1 · Tiempo de inserción vs Nº de conceptos')
    ax.set_xlabel('Conceptos insertados')
    ax.set_ylabel('Segundos')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, 'b1_scalability.png'), dpi=120)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# B2 – Latencia de retrieve (Spreading Activation)
# ─────────────────────────────────────────────────────────────────────────────
def b2_retrieve_latency():
    print("\n[B2] Latencia de Retrieve / Spreading Activation")
    clean('b2')
    cfg = AESGConfig(vector_dim=16, max_concepts=2000, novelty_birth_threshold=1)
    mem = AESGMemory(tmp_dir('b2'), config=cfg)

    # Poblar con 1000 conceptos
    for _ in range(1000):
        mem.cognitive_engine.create_sensory_concept(torch.randn(16))

    hops_list = [1, 2, 4, 8]
    results = []

    for hops in hops_list:
        mem.config.spreading_activation_steps = hops
        latencies = []
        for _ in range(200):
            q = torch.randn(16)
            mem.reset_state()
            t0 = time.perf_counter()
            mem.retrieve(q)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg  = round(float(np.mean(latencies)), 3)
        p99  = round(float(np.percentile(latencies, 99)), 3)
        results.append({'hops': hops, 'avg_ms': avg, 'p99_ms': p99})
        print(f"  {hops} hops → avg={avg:.2f}ms  p99={p99:.2f}ms")

    DB['b2_latency'] = results
    safe_del(mem)
    clean('b2')

    fig, ax = plt.subplots()
    ax.plot([r['hops'] for r in results], [r['avg_ms'] for r in results],
            marker='o', label='Media', color='steelblue')
    ax.plot([r['hops'] for r in results], [r['p99_ms'] for r in results],
            marker='x', linestyle='--', label='P99', color='tomato')
    ax.set_title('B2 · Latencia de Retrieve vs Hops')
    ax.set_xlabel('Spreading Activation Hops')
    ax.set_ylabel('Latencia (ms)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, 'b2_latency.png'), dpi=120)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# B3 – Estrés de Consolidación y Presión Evolutiva
# ─────────────────────────────────────────────────────────────────────────────
def b3_evolutionary_pressure():
    print("\n[B3] Presión Evolutiva")
    clean('b3')
    MAX = 3000
    cfg = AESGConfig(
        vector_dim=16, max_concepts=MAX,
        survival_threshold_relevance=0.3,
        novelty_birth_threshold=1
    )
    mem = AESGMemory(tmp_dir('b3'), config=cfg)

    t0 = time.perf_counter()
    INSERTED = 10_000
    for i in range(INSERTED):
        mem.cognitive_engine.create_sensory_concept(torch.randn(16))
        if i % 2000 == 0:
            mem.cognitive_engine.consolidate()

    elapsed = time.perf_counter() - t0
    active = int(np.sum(mem.storage.nodes['is_active'][:mem.storage.node_count]))
    pruned = INSERTED - active

    print(f"  Insertados={INSERTED}, MAX={MAX}, Activos={active}, Podados≈{pruned}, t={elapsed:.2f}s")
    DB['b3_pressure'] = {
        'inserted': INSERTED, 'max_allowed': MAX,
        'survivors': active, 'approx_pruned': pruned,
        'time_s': round(elapsed, 3)
    }

    safe_del(mem)
    clean('b3')

# ─────────────────────────────────────────────────────────────────────────────
# B4 – Curiosidad Adaptativa (filtro de novedad)
# ─────────────────────────────────────────────────────────────────────────────
def b4_adaptive_curiosity():
    print("\n[B4] Curiosidad Adaptativa")
    clean('b4')
    cfg = AESGConfig(
        vector_dim=16,
        novelty_birth_threshold=4,
        novelty_explanation_threshold=0.85
    )
    mem = AESGMemory(tmp_dir('b4'), config=cfg)

    # Ruido puro → no debe crear conceptos más allá del bootstrap
    for _ in range(8):
        mem.retrieve(torch.randn(16))
    after_noise = mem.storage.node_count

    # Patrón repetido → debe crear exactamente 1 concepto nuevo
    pattern = torch.randn(16)
    for _ in range(5):
        mem.retrieve(pattern + torch.randn(16) * 0.01)
    after_pattern = mem.storage.node_count

    print(f"  Conceptos tras ruido={after_noise}, tras patrón={after_pattern}")
    DB['b4_curiosity'] = {
        'concepts_after_noise': after_noise,
        'concepts_after_pattern': after_pattern,
        'new_concepts_from_pattern': after_pattern - after_noise
    }

    safe_del(mem)
    clean('b4')

# ─────────────────────────────────────────────────────────────────────────────
# B5 – Evolution Log (I/O binario)
# ─────────────────────────────────────────────────────────────────────────────
def b5_evolution_log():
    print("\n[B5] Evolution Log")
    clean('b5')
    d = tmp_dir('b5')
    logger = EvolutionLogger(d)
    N = 100_000

    t0 = time.perf_counter()
    for i in range(N):
        logger.log_event("CREATE", id1=i)
    write_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    hist = logger.get_history()
    read_t = time.perf_counter() - t0

    size_mb = os.path.getsize(logger.log_path) / 1024**2
    print(f"  {N} eventos: escritura={write_t:.3f}s  lectura={read_t:.3f}s  size={size_mb:.2f}MB")
    DB['b5_log'] = {
        'events': N, 'write_s': round(write_t, 4),
        'read_s': round(read_t, 4), 'size_mb': round(size_mb, 3)
    }
    clean('b5')

# ─────────────────────────────────────────────────────────────────────────────
# B6 – Stress RAM y Disco (1M conceptos, dim=8)
# ─────────────────────────────────────────────────────────────────────────────
def b6_stress():
    print("\n[B6] Stress Test (1M conceptos, dim=8)")
    clean('b6')
    N = 1_000_000
    cfg = AESGConfig(vector_dim=8, max_concepts=N + 1000)
    mem = AESGMemory(tmp_dir('b6'), config=cfg)

    r0 = ram_mb()
    t0 = time.perf_counter()
    vecs = torch.randn(N, 8)
    for i in range(N):
        mem.cognitive_engine.create_sensory_concept(vecs[i])
    elapsed = time.perf_counter() - t0
    r1 = ram_mb()

    mem.save()
    disk = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(tmp_dir('b6')) for f in fs
    ) / 1024**2

    print(f"  1M conceptos: t={elapsed:.2f}s, ΔRAM={r1-r0:.1f}MB, disco={disk:.1f}MB")
    DB['b6_stress'] = {
        'n': N, 'time_s': round(elapsed, 3),
        'ram_delta_mb': round(r1 - r0, 1),
        'disk_mb': round(disk, 2)
    }

    safe_del(mem)
    clean('b6')

# ─────────────────────────────────────────────────────────────────────────────
# Generación del informe final
# ─────────────────────────────────────────────────────────────────────────────
def generate_report():
    # JSON raw
    with open(os.path.join(RESULTS, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(DB, f, indent=2, ensure_ascii=False)

    # Markdown
    lines = ["# AESG V2.1 — Informe de Benchmarks\n",
             f"*Ejecutado: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
             "---\n"]

    # B1
    lines.append("## B1 · Escalabilidad de Inserción\n")
    lines.append("| Conceptos | Tiempo (s) | Ops/s | Disco (MB) |")
    lines.append("|----------:|-----------:|------:|-----------:|")
    for r in DB.get('b1_scalability', []):
        lines.append(f"| {r['n']:,} | {r['time_s']} | {r['ops_s']:,.0f} | {r['disk_mb']} |")

    lines.append("\n![B1 Escalabilidad](../../docs/images/b1_scalability.png)\n")

    # B2
    lines.append("## B2 · Latencia de Retrieve\n")
    lines.append("| Hops | Avg (ms) | P99 (ms) |")
    lines.append("|-----:|---------:|---------:|")
    for r in DB.get('b2_latency', []):
        lines.append(f"| {r['hops']} | {r['avg_ms']} | {r['p99_ms']} |")
    lines.append("\n![B2 Latencia](../../docs/images/b2_latency.png)\n")

    # B3
    p = DB.get('b3_pressure', {})
    lines.append("## B3 · Presión Evolutiva\n")
    lines.append(f"- **Insertados:** {p.get('inserted','-'):,}")
    lines.append(f"- **Límite activo (max_concepts):** {p.get('max_allowed','-'):,}")
    lines.append(f"- **Supervivientes:** {p.get('survivors','-'):,}")
    lines.append(f"- **Tiempo total:** {p.get('time_s','-')} s\n")

    # B4
    c = DB.get('b4_curiosity', {})
    lines.append("## B4 · Curiosidad Adaptativa\n")
    lines.append(f"- Conceptos tras **ruido puro** (8 passes): **{c.get('concepts_after_noise','-')}**")
    lines.append(f"- Conceptos tras **patrón repetido** (5 passes): **{c.get('concepts_after_pattern','-')}**")
    lines.append(f"- Nuevos conceptos aprendidos del patrón: **{c.get('new_concepts_from_pattern','-')}**\n")

    # B5
    lg = DB.get('b5_log', {})
    lines.append("## B5 · Evolution Log Binario\n")
    lines.append(f"- **100k eventos escritura:** {lg.get('write_s','-')} s")
    lines.append(f"- **100k eventos lectura:** {lg.get('read_s','-')} s")
    lines.append(f"- **Tamaño en disco:** {lg.get('size_mb','-')} MB\n")

    # B6
    st = DB.get('b6_stress', {})
    lines.append("## B6 · Stress Test Extremo (1M conceptos, dim=8)\n")
    lines.append(f"- **Tiempo de inserción:** {st.get('time_s','-')} s")
    lines.append(f"- **Overhead RAM:** {st.get('ram_delta_mb','-')} MB")
    lines.append(f"- **Tamaño en disco:** {st.get('disk_mb','-')} MB\n")

    md = "\n".join(lines)
    with open(os.path.join(RESULTS, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n✓ Informe guardado en {RESULTS}")
    print(f"✓ Gráficas guardadas en {IMAGES}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  AESG V2.1 · Suite de Benchmarks Científicos")
    print("=" * 60)

    b1_scalability()
    b2_retrieve_latency()
    b3_evolutionary_pressure()
    b4_adaptive_curiosity()
    b5_evolution_log()
    b6_stress()

    generate_report()
    print("\n✅ TODOS LOS BENCHMARKS COMPLETADOS.")
    if os.path.exists(TMP_BASE):
        shutil.rmtree(TMP_BASE, ignore_errors=True)
