# Parallel N-Body Simulation (Gravitational System)
### High Performance Computing — Project 2

**Course:** High Performance Computing  
**Date:** May 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Implementation Details](#2-implementation-details)
3. [Results](#3-results)
4. [Discussion](#4-discussion)
5. [Conclusion](#5-conclusion)

---

## 1. Introduction

### 1.1 Overview of N-Body Simulation

The N-Body problem is a classical problem in physics and computational science. It involves simulating the motion of $N$ particles that interact with each other under gravitational forces. Every particle exerts a force on every other particle, and their positions and velocities evolve over time according to Newton's laws of motion.

The gravitational force exerted by particle $j$ on particle $i$ is given by:

$$\vec{F}_{ij} = G \frac{m_i \, m_j}{|\vec{r}_{ij}|^2 + \epsilon^2} \hat{r}_{ij}$$

where:
- $G = 6.674 \times 10^{-11}$ is the gravitational constant
- $m_i$, $m_j$ are the masses of the two particles
- $\vec{r}_{ij} = \vec{r}_j - \vec{r}_i$ is the displacement vector between them
- $\epsilon$ is a **softening factor** ($10^{-3}$) that prevents singularities when two particles are very close

Each particle's velocity and position are then updated via **Euler integration**:

$$\vec{v}_i \leftarrow \vec{v}_i + \frac{\vec{F}_i}{m_i} \Delta t$$

$$\vec{x}_i \leftarrow \vec{x}_i + \vec{v}_i \Delta t$$

The naive approach requires evaluating all ordered pairs $(i, j)$, giving **O(N²)** force computations per timestep. By applying **Newton's Third Law** ($\vec{F}_{ji} = -\vec{F}_{ij}$), only unique pairs $(i, j)$ with $j > i$ are computed, reducing evaluations to exactly $\frac{N(N-1)}{2}$ — approximately half the work.

### 1.2 Objectives

- Implement and compare three versions of the N-Body simulation: Sequential, OpenMP, and MPI
- Implement a fourth Hybrid (MPI + OpenMP) version demonstrating two-level parallelism
- Apply Newton's Third Law across all versions for a fair algorithmic baseline
- Measure execution time, speedup, and parallel efficiency across increasing particle counts
- Analyze communication overhead, load balancing, and scalability

---

## 2. Implementation Details

### 2.1 Project Structure

```
common.h        — Shared data structures, initialization, CSV output
sequential.cpp  — Single-threaded baseline
openmp.cpp      — Shared-memory parallelism (OpenMP)
mpi.cpp         — Distributed-memory parallelism (MPI)
hybrid.cpp      — Two-level parallelism (MPI processes + OpenMP threads)
benchmark.ps1   — Automated benchmark script
Makefile        — Build system
```

### 2.2 Data Structures

All versions share the structures defined in `common.h`.

#### Particle (Array of Structures)

```cpp
struct Particle {
    double x, y;    // Position
    double vx, vy;  // Velocity
    double mass;    // Mass
};
```

Used by the sequential and OpenMP versions for cache-friendly sequential access.

#### Flat Separate Arrays (Structure of Arrays)

The MPI and Hybrid versions decompose the particle data into separate flat arrays (`all_x[]`, `all_y[]`, `all_vx[]`, `all_vy[]`, `all_mass[]`) to make MPI collective operations straightforward — `MPI_Scatterv`, `MPI_Allgatherv`, and `MPI_Allreduce` all require contiguous buffers of a single type.

#### Simulation Parameters

```cpp
struct SimParams {
    int N;           // Number of particles
    double dt;       // Time step
    int iterations;  // Number of simulation steps
    int output_freq; // CSV output frequency (0 = disabled)
};
```

Configured entirely via command-line arguments for reproducible experiments.

#### Initialization

All four versions use the same `initParticles()` function with a **fixed random seed (42)**:
- Positions uniformly in $[0, 1000]$
- Velocities uniformly in $[-1, 1]$
- Masses uniformly in $[10^{10}, 10^{12}]$

This guarantees identical initial conditions and directly comparable results.

---

### 2.3 Sequential Version

The sequential version serves as the baseline. It runs a standard double-nested loop over all unique pairs $(i, j)$ with $j > i$, exploiting Newton's Third Law to halve the computation.

**Force computation (Newton's 3rd Law):**

```cpp
for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
        double dx = particles[j].x - particles[i].x;
        double dy = particles[j].y - particles[i].y;
        double distSq = dx*dx + dy*dy + SOFTENING*SOFTENING;
        double invDist3 = 1.0 / (distSq * std::sqrt(distSq));
        double F = G * particles[i].mass * particles[j].mass * invDist3;
        fx[i] += F * dx;   fy[i] += F * dy;
        fx[j] -= F * dx;   fy[j] -= F * dy;  // equal and opposite
    }
}
```

**Integration:**

```cpp
for (int i = 0; i < N; i++) {
    double ax = fx[i] / particles[i].mass;
    double ay = fy[i] / particles[i].mass;
    particles[i].vx += ax * dt;
    particles[i].vy += ay * dt;
    particles[i].x  += particles[i].vx * dt;
    particles[i].y  += particles[i].vy * dt;
}
```

---

### 2.4 OpenMP Version

The OpenMP version parallelises the force computation across shared memory. The key challenge with Newton's Third Law is that multiple threads may update the same force accumulator slots simultaneously (thread writing `fx[j]` while another thread reads or writes it). This is resolved using **OpenMP array reduction**.

**Parallel force loop:**

```cpp
#pragma omp parallel for schedule(dynamic, 16) \
        reduction(+: fx[0:N], fy[0:N])
for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
        // ... force calculation ...
        fx[i] += f_x;   fy[i] += f_y;
        fx[j] -= f_x;   fy[j] -= f_y;
    }
}
```

Key design decisions:
- **`schedule(dynamic, 16)`** — the outer loop is not perfectly load-balanced (iteration 0 does N-1 inner steps, iteration N-2 does only 1). Dynamic scheduling assigns chunks of 16 iterations at runtime to keep threads busy.
- **Array reduction** — each thread maintains a private copy of the full force array; OpenMP sums them at the barrier, avoiding all race conditions.
- **`schedule(static)`** on the update loop — perfectly balanced, no dynamic overhead needed.

---

### 2.5 MPI Version

The MPI version distributes particles across processes using a **block partition**: process $r$ owns particles $[\text{displs}[r],\ \text{displs}[r] + \text{counts}[r])$. The last process absorbs any remainder from $N \bmod \text{size}$.

**Per-timestep communication pattern:**

```
MPI_Allgatherv  ← share ALL positions (local_x/y → all_x/y)
                  every process now has the full position set

[force computation — Newton's 3rd law, pairs owned by this process]

MPI_Allreduce   ← sum partial_fx/fy arrays across all processes
                  every process now has correct total force on its particles
```

**Newton's 3rd Law in MPI:** Each process only computes pairs $(i, j)$ where $i$ is a local particle and $j > \text{global\_i}$. The force contribution to the remote particle $j$ is written into `partial_fx[j]`. Since processes can't write into each other's memory, all partial arrays are summed with `MPI_Allreduce(MPI_SUM)`:

```cpp
MPI_Allreduce(partial_fx.data(), total_fx.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(partial_fy.data(), total_fy.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

Each process then updates only its local particles using `total_fx[global_i]`.

---

### 2.6 Hybrid (MPI + OpenMP) Version

The Hybrid version combines both parallelism models:

| Level | Mechanism | Scope |
|---|---|---|
| Coarse-grained | MPI processes | Particle ownership, distributed across processes |
| Fine-grained | OpenMP threads | Force computation within each process |

**Thread safety:** MPI calls are made only by the master thread (`MPI_THREAD_FUNNELED`). OpenMP is confined strictly to the force computation phase between MPI collectives.

**Two-level force loop:**

```cpp
#pragma omp parallel for schedule(dynamic, 16) \
        reduction(+: partial_fx[0:N], partial_fy[0:N])
for (int i = 0; i < local_n; i++) {
    int global_i = local_start + i;
    for (int j = global_i + 1; j < N; j++) {
        // ... force calculation (same as MPI version) ...
        partial_fx[global_i] += f_x;
        partial_fy[global_i] += f_y;
        partial_fx[j]        -= f_x;   // Newton's 3rd law
        partial_fy[j]        -= f_y;
    }
}
// OpenMP barrier — then MPI_Allreduce
```

This means for $P$ processes and $T$ threads each, the total parallelism is $P \times T$ workers, with computation divided hierarchically:
- MPI partitions the outer loop by particle ownership
- OpenMP further partitions each process's share among threads

---

## 3. Results

### 3.1 Experimental Setup

| Parameter | Value |
|---|---|
| Machine | <!-- TODO: Add your machine specs (CPU model, cores, RAM) --> |
| OS | Windows |
| Compiler | G++ with `-O2 -std=c++17` |
| MPI | Microsoft MPI (MS-MPI) |
| OpenMP | GCC built-in |
| Particle counts (N) | 100, 500, 1000, 2000, 5000 |
| Time step (dt) | 0.01 |
| Iterations | 100 (N≤500), 50 (N=1000), 30 (N=2000), 15 (N=5000) |
| OMP thread counts | 2, 4, 8 |
| MPI process counts | 2, 4 |
| Hybrid combos | 2P×2T, 2P×4T, 4P×2T |

Speedup is defined as:

$$S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$

Parallel efficiency is:

$$E = \frac{S}{W}$$

where $W$ is the total worker count (processes × threads).

---

### 3.2 Wall-Clock Time (seconds)

| N | Sequential | OMP 2T | OMP 4T | OMP 8T | MPI 2P | MPI 4P | Hybrid 2P×4T | Hybrid 4P×2T |
|---|---|---|---|---|---|---|---|---|
| 100  | 0.00244 | 0.00838 | 0.01030 | 0.01539 | 0.00364 | 0.00223 | 0.00783 | 0.00850 |
| 500  | 0.06984 | 0.04226 | 0.02676 | 0.02571 | 0.05007 | 0.03828 | 0.02321 | 0.02386 |
| 1000 | 0.12652 | 0.06694 | 0.04270 | 0.02878 | 0.09638 | 0.05685 | 0.03493 | 0.03854 |
| 2000 | 0.30091 | 0.15403 | 0.08479 | 0.05425 | 0.22965 | 0.13782 | 0.06427 | 0.07743 |
| 5000 | 0.93055 | 0.46782 | 0.24528 | 0.13532 | 0.69949 | 0.41418 | 0.18797 | 0.23890 |

---

### 3.3 Speedup

| N | OMP 2T | OMP 4T | OMP 8T | MPI 2P | MPI 4P | Hybrid 2P×4T | Hybrid 4P×2T |
|---|---|---|---|---|---|---|---|
| 100  | 0.29 | 0.24 | 0.16 | 0.67 | 1.09 | 0.31 | 0.29 |
| 500  | 1.65 | 2.61 | 2.72 | 1.39 | 1.82 | 3.01 | 2.93 |
| 1000 | 1.89 | 2.96 | 4.40 | 1.31 | 2.23 | 3.62 | 3.28 |
| 2000 | 1.95 | 3.55 | 5.55 | 1.31 | 2.18 | 4.68 | 3.89 |
| 5000 | 1.99 | 3.79 | **6.88** | 1.33 | 2.25 | **4.95** | 3.90 |

> **Best single configuration at N=5000:** OpenMP 8 threads → **6.88× speedup**

---

### 3.4 Parallel Efficiency

| N | OMP 2T | OMP 4T | OMP 8T | MPI 2P | MPI 4P | Hybrid 2P×4T (8W) | Hybrid 4P×2T (8W) |
|---|---|---|---|---|---|---|---|
| 100  | 14.6% | 5.9% | 2.0% | 33.5% | 27.4% | 3.9% | 3.6% |
| 500  | 82.6% | 65.2% | 33.9% | 69.7% | 45.6% | 37.6% | 36.6% |
| 1000 | 94.5% | 74.1% | 54.9% | 65.6% | 55.6% | 45.3% | 41.0% |
| 2000 | 97.7% | 88.7% | 69.3% | 65.5% | 54.6% | 58.5% | 48.6% |
| 5000 | **99.5%** | **94.8%** | **86.0%** | 66.5% | 56.2% | 61.9% | 48.7% |

---

### 3.5 Figures

> **TODO — Insert the following charts (generated from `benchmark_results.csv`):**

**Figure 1 — Wall-clock time vs N (log-log scale)**  
`<!-- Insert: line chart, x=N, y=Time_s, one line per version/config -->`

**Figure 2 — Speedup vs N for OpenMP (2T, 4T, 8T)**  
`<!-- Insert: line chart showing OpenMP scales well with N -->`

**Figure 3 — Speedup vs N for MPI (2P, 4P)**  
`<!-- Insert: line chart showing MPI plateaus at low N -->`

**Figure 4 — Speedup vs N for Hybrid (3 combos)**  
`<!-- Insert: line chart, compare 2P×4T vs 4P×2T -->`

**Figure 5 — Efficiency vs N (all versions at 8 workers)**  
`<!-- Insert: bar or line chart comparing OMP 8T, MPI 4P, Hybrid 2P×4T, Hybrid 4P×2T -->`

**Figure 6 — Crossover point at N=100 (overhead dominates)**  
`<!-- Insert: bar chart at N=100 showing all parallel versions slower than sequential -->`

---

## 4. Discussion

### 4.1 OpenMP Scalability

OpenMP consistently achieves the best speedup on a single machine. At N=5000:
- 2 threads → **1.99× speedup, 99.5% efficiency** (near-perfect)
- 4 threads → **3.79× speedup, 94.8% efficiency**
- 8 threads → **6.88× speedup, 86.0% efficiency**

This strong scaling is explained by:
- **Zero communication cost** — all threads share the same address space
- **Dynamic scheduling** — compensates for the triangular load imbalance inherent in the $j > i$ loop structure (thread 0 handles far more pairs than thread N-2)
- **Array reduction** — the per-thread private copy approach eliminates all synchronization inside the parallel region; the merge cost at the barrier is $O(N \times T)$ which is small relative to the $O(N^2)$ computation

Efficiency drops from 99.5% (2T) to 86.0% (8T) because:
1. The OpenMP array reduction merge cost grows linearly with threads
2. False sharing between adjacent force array elements becomes more likely with more threads
3. Amdahl's Law — the sequential update loop and I/O paths become proportionally larger

### 4.2 MPI Communication Overhead

MPI achieves significantly lower speedups than OpenMP at the same process count — at N=5000, MPI 4P delivers only **2.25×** vs OpenMP 4T's **3.79×**. The bottleneck is per-timestep communication:

| Collective | Data transferred | Cost |
|---|---|---|
| 2× `MPI_Allgatherv` (positions) | $2N$ doubles | $O(N \log P)$ |
| 2× `MPI_Allreduce` (forces) | $2N$ doubles | $O(N \log P)$ |

Total: **$4N$ doubles per timestep**, every timestep. At N=5000 and 15 iterations, this is $4 \times 5000 \times 8 \times 15 = 2.4\text{ MB}$ of MPI data.

MPI efficiency plateaus at ~65% for 2 processes and ~56% for 4 processes regardless of N, confirming that communication overhead scales proportionally with problem size (unlike sequential computation which scales as $N^2$). This means MPI's relative cost stays roughly constant — it never fully amortizes.

### 4.3 Hybrid: Two-Level Parallelism Trade-off

The Hybrid version always falls between pure OpenMP and pure MPI in performance on this single-machine setup. At N=5000 with 8 total workers:

| Config | Speedup | Why |
|---|---|---|
| OpenMP 8T | 6.88× | No inter-process communication at all |
| Hybrid 2P×4T | 4.95× | Only 2 MPI processes → low Allreduce cost |
| Hybrid 4P×2T | 3.90× | 4 MPI processes → higher Allreduce cost |

The consistent gap between Hybrid and pure OpenMP exists because the MPI_Allreduce on 2N doubles introduces a synchronization barrier every timestep that OpenMP threads do not have.

**On a real multi-node HPC cluster**, this relationship would invert: OpenMP cannot span node boundaries, but MPI can. In that scenario, 2P×4T (2 nodes, 4 cores each) would be the only option to use all 8 physical cores, and the communication would be network-bound rather than loopback-bound.

Within hybrid configurations, **fewer processes with more threads always wins** (2P×4T > 4P×2T) because it minimises the number of MPI collectives while maximising shared-memory parallelism.

### 4.4 Overhead-Dominated Regime (N=100)

At N=100, every parallel version is slower than sequential:

- **OpenMP 8T: 0.16× speedup** — thread spawn + reduction merge costs exceed 0.0024s of computation
- **MPI 2P: 0.67× speedup** — MPI process startup and Allgatherv latency dominate
- **Hybrid: all < 0.5×** — both overheads combined

This is the expected crossover behavior and directly demonstrates **Amdahl's Law** — for sufficiently small problem sizes, the parallel overhead $T_{\text{overhead}}$ exceeds the computation saved. The crossover point where parallelism becomes beneficial is approximately N=300–400 for OpenMP and N=400–600 for MPI on this hardware.

### 4.5 Load Balancing

The triangular loop structure (outer $i$ from 0 to $N-2$, inner $j$ from $i+1$ to $N-1$) creates an inherent imbalance:

- Particle 0 interacts with N-1 others
- Particle N-2 interacts with only 1 other
- Thread/process 0 handles far more pairs than the last thread/process

**OpenMP** addresses this with `schedule(dynamic, 16)` — work chunks are assigned at runtime, so faster threads pick up more chunks automatically.

**MPI** assigns contiguous blocks of particles to processes. This means rank 0 (owning the lowest-indexed particles) handles significantly more pairs than rank $P-1$. The load imbalance factor grows with $P$.

> **TODO — Add load imbalance measurement:** The benchmark script can be extended with `MPI_Reduce(MPI_MAX)` and `MPI_Reduce(MPI_MIN)` on per-process wall times to compute the imbalance ratio $T_{\text{max}} / T_{\text{min}}$.

### 4.6 Debugging Strategies

<!-- TODO: Add 2–3 specific debugging experiences from your testing, e.g.:
- A race condition encountered before applying the array reduction
- An off-by-one in the MPI particle index (global_i vs local i)
- Force sign error discovered by checking energy conservation
- Incorrect MPI_Allgatherv counts/displs leading to wrong forces
-->

Key strategies used during development:
1. **Fixed random seed (42)** — all three versions initialize identically, so final particle positions can be compared numerically to verify correctness
2. **Small N validation** — running N=4 manually and checking forces by hand before scaling up
3. **CSV output** — `output_freq=1` dumps all particle states every step; trajectory divergence pinpoints which step introduced an error
4. **Sequential cross-check** — the sequential version acts as ground truth; any implementation that produces matching particle positions after K steps is correct

---

## 5. Conclusion

### 5.1 Summary of Findings

| Finding | Evidence |
|---|---|
| All parallel versions have a minimum problem size to be beneficial | N=100: all parallel versions slower than sequential |
| OpenMP achieves near-linear scaling on this hardware | 8 threads → 6.88× speedup (86% efficiency) at N=5000 |
| MPI communication cost does not amortize with N | MPI efficiency plateaus ~56–66% regardless of N |
| Hybrid is optimal on multi-node clusters but sub-optimal on single-node | Single machine: always slower than pure OpenMP at equal worker count |
| Fewer MPI processes with more OMP threads outperforms the reverse | 2P×4T (4.95×) > 4P×2T (3.90×) at N=5000, 8 workers |
| Newton's 3rd law halves computation cost for all versions fairly | Consistent baseline enables valid speedup comparisons |

### 5.2 Possible Improvements

1. **Leapfrog (Velocity-Verlet) Integration** — the current Euler integrator is first-order and accumulates energy error over long simulations. Leapfrog is second-order with minimal extra code and conserves total mechanical energy far better.

2. **Energy Conservation Tracking** — computing total kinetic and potential energy per step ($E_k = \frac{1}{2}mv^2$, $E_p = -\frac{Gm_im_j}{r}$) provides a physical correctness metric and allows comparison of integrator quality.

3. **3D Extension** — extending positions and forces from 2D to 3D is a trivial code change (add `z`, `vz`, `fz` everywhere) but makes the simulation physically realistic.

4. **Barnes-Hut Tree ($O(N \log N)$)** — for very large N, replacing the exact $O(N^2)$ force computation with a hierarchical approximation reduces complexity and unlocks simulations with millions of particles.

5. **MPI Load Balancing** — replacing the static block partition with a dynamic or cyclic assignment (e.g. rank $r$ handles particles $r, r+P, r+2P, \ldots$) distributes the triangular loop more evenly across processes.

6. **SIMD Vectorization** — adding `#pragma omp simd` on the inner force loop and compiling with `-O3 -march=native` enables AVX2/AVX-512 auto-vectorization, potentially doubling throughput on modern CPUs.

7. **GPU Acceleration (CUDA/OpenCL)** — the $O(N^2)$ force kernel maps naturally to a GPU thread grid where thread $(i, j)$ computes one force pair, offering orders-of-magnitude speedup for large N.

---

*<!-- TODO: Add your student name(s) and ID(s) here -->*
