# AESG V2.1 Benchmark Results

## 1. Escalabilidad de Conceptos
- 10000 Nodos: 2.74s (3644 ops/s) | RAM: 2.12MB | Disco: 3.42MB
- 100000 Nodos: 38.86s (2573 ops/s) | RAM: 17.29MB | Disco: 31.66MB
- 500000 Nodos: 220.05s (2272 ops/s) | RAM: 79.57MB | Disco: 151.40MB
- 1000000 Nodos: 539.27s (1854 ops/s) | RAM: 172.88MB | Disco: 314.04MB

## 2. Escalabilidad de Navegación (Spreading Activation)
- 2 Hops: Media 3.72ms | p99 6.90ms
- 4 Hops: Media 8.99ms | p99 20.45ms
- 8 Hops: Media 48.58ms | p99 139.30ms
- 16 Hops: Media 163.19ms | p99 714.61ms

## 4. Presión Evolutiva
Insertados 15000, Límite 5000. Supervivientes: 5999, Eliminados: 0.

## 5. Curiosidad Adaptativa
Ruido insertado: 10 veces -> Conceptos en grafo: 4
Patrón insertado: 4 veces -> Conceptos en grafo: 5 (Creación validada).

## 8. Evolution Log (Binario)
100k Eventos: Escritura 41.89s | Lectura 0.13s | Disco 3.05MB.

## 10. Stress Test Extremo
Crecimiento máximo exitoso: 1000000 conceptos.
Tiempo total inyección: 451.87s.
Consumo Pico RAM: 406.96MB. Tamaño Disco: 314.04MB.
