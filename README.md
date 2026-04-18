# Parallel N-Body Simulation (2D Gravitational System)

A 2D gravitational N-Body simulation with three implementations:
1. **Sequential** — baseline single-threaded version
2. **OpenMP** — shared-memory parallelism across CPU threads
3. **MPI** — distributed-memory parallelism across processes

Each version computes O(N²) pairwise gravitational forces per timestep and updates particle positions/velocities via Euler integration.

## Project Structure

```
common.h          Shared: Particle struct, initialization, CSV writer, arg parser
sequential.cpp    Sequential double-loop implementation
openmp.cpp        OpenMP-parallelized force computation
mpi.cpp           MPI-distributed force computation with Allgatherv
Makefile          Build targets for all three versions
README.md         This file
```

## Build

Requires: `g++` (with C++17 and OpenMP support) and `mpicxx` (MPI compiler wrapper).

```bash
make all          # Build all three versions
make sequential   # Build only sequential
make openmp       # Build only OpenMP
make mpi          # Build only MPI
make clean        # Remove executables and CSV output
```

## Usage

All three programs accept the same command-line arguments:

```
./program <N> <dt> <iterations> [output_freq]
```

| Argument      | Description                              | Default |
|---------------|------------------------------------------|---------|
| `N`           | Number of particles                      | 500     |
| `dt`          | Time step size                           | 0.01    |
| `iterations`  | Number of simulation steps               | 100     |
| `output_freq` | Write CSV every N steps (0 = no output)  | 0       |

### Examples

```bash
# Sequential: 1000 particles, dt=0.01, 100 iterations, CSV every 10 steps
./sequential 1000 0.01 100 10

# OpenMP: 2000 particles with 4 threads
OMP_NUM_THREADS=4 ./openmp 2000 0.01 100

# MPI: 2000 particles across 4 processes
mpirun -np 4 ./mpi 2000 0.01 100
```

## Output

- **Timing**: Wall-clock time and interactions/sec printed to stdout
- **CSV files** (when `output_freq > 0`):
  - `output_sequential.csv`
  - `output_openmp.csv`
  - `output_mpi.csv`
  - Format: `step, id, x, y, vx, vy`

## Algorithm

1. Initialize N particles with random positions, velocities, and masses (fixed seed = 42)
2. For each timestep:
   - Compute gravitational force on particle i from all other particles j ≠ i
   - `F = G * m_i * m_j / (|r_ij|² + ε²)^(3/2)` (softened to prevent singularity)
   - Update velocity: `v += (F/m) * dt`
   - Update position: `x += v * dt`
3. Repeat for the specified number of iterations

### Parallelization Strategy

- **OpenMP**: The outer loop over particles is parallelized with `#pragma omp parallel for schedule(dynamic)`. Each thread computes forces for its assigned particles independently — no race conditions since force accumulators are per-particle.

- **MPI**: Particles are distributed across processes. Each timestep, all processes exchange positions via `MPI_Allgatherv` so every process has the full particle set for force computation. Each process computes forces only for its local particle subset.

## Performance Analysis

Compare wall-clock times across implementations:

```bash
# Sequential baseline
./sequential 2000 0.01 50

# OpenMP scaling
OMP_NUM_THREADS=1 ./openmp 2000 0.01 50
OMP_NUM_THREADS=2 ./openmp 2000 0.01 50
OMP_NUM_THREADS=4 ./openmp 2000 0.01 50
OMP_NUM_THREADS=8 ./openmp 2000 0.01 50

# MPI scaling
mpirun -np 1 ./mpi 2000 0.01 50
mpirun -np 2 ./mpi 2000 0.01 50
mpirun -np 4 ./mpi 2000 0.01 50
```

**Speedup** = T_sequential / T_parallel

**Efficiency** = Speedup / Number_of_threads_or_processes
