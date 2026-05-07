// ============================================================
// Hybrid MPI + OpenMP N-Body Simulation (2D Gravitational)
// ============================================================
// Combines two levels of parallelism:
//
//   MPI  — distributes particle ownership across processes.
//           Uses MPI_Allgatherv to share positions each step,
//           and MPI_Allreduce to sum Newton's 3rd law partial
//           force contributions across all processes.
//
//   OpenMP — parallelises the force loop WITHIN each MPI process
//            using an array reduction so multiple threads can
//            safely accumulate into the shared partial_fx/fy
//            arrays without race conditions.
//
// Newton's 3rd law: only pairs (global_i, j) with j > global_i
// are computed — by the process that owns particle i.  This
// halves the computation cost vs a naive full-loop approach.
//
// Thread safety: MPI calls are made only by the master thread
// (MPI_THREAD_FUNNELED), so OpenMP is confined to the pure
// compute phases between MPI collectives.
// ============================================================

#include "common.h"
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {

    // Request thread support: only the master thread calls MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parse arguments ----
    SimParams params = parseArgs(argc, argv);
    int N           = params.N;
    double dt       = params.dt;
    int iterations  = params.iterations;
    int output_freq = params.output_freq;

    int num_threads = omp_get_max_threads();

    if (rank == 0) {
        std::cout << "=== Hybrid MPI+OpenMP N-Body Simulation ===\n";
        std::cout << "Particles     : " << N << "\n";
        std::cout << "Timestep      : " << dt << "\n";
        std::cout << "Iterations    : " << iterations << "\n";
        std::cout << "MPI Processes : " << size << "\n";
        std::cout << "OMP Threads   : " << num_threads << "\n";
        std::cout << "Total workers : " << (size * num_threads) << "\n\n";
    }

    // ---- Determine how particles are distributed ----
    // Same block distribution as the MPI version so results are
    // comparable. The last process absorbs any remainder.
    std::vector<int> counts(size), displs(size);
    int base      = N / size;
    int remainder = N % size;
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }
    int local_n     = counts[rank];
    int local_start = displs[rank];

    // ---- Initialize all particles on rank 0 and scatter ----
    std::vector<double> all_x(N), all_y(N);
    std::vector<double> all_vx(N), all_vy(N);
    std::vector<double> all_mass(N);

    if (rank == 0) {
        std::vector<Particle> temp;
        initParticles(N, temp);
        for (int i = 0; i < N; i++) {
            all_x[i]    = temp[i].x;
            all_y[i]    = temp[i].y;
            all_vx[i]   = temp[i].vx;
            all_vy[i]   = temp[i].vy;
            all_mass[i] = temp[i].mass;
        }
    }

    // Broadcast masses once — they are constant throughout
    MPI_Bcast(all_mass.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter initial positions and velocities
    std::vector<double> local_x(local_n),  local_y(local_n);
    std::vector<double> local_vx(local_n), local_vy(local_n);

    MPI_Scatterv(all_x.data(),  counts.data(), displs.data(), MPI_DOUBLE,
                 local_x.data(),  local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_y.data(),  counts.data(), displs.data(), MPI_DOUBLE,
                 local_y.data(),  local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vx.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local_vx.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local_vy.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Write initial state on rank 0
    std::string csvFile = "output_hybrid.csv";
    if (output_freq > 0 && rank == 0) {
        std::vector<Particle> all_particles(N);
        for (int i = 0; i < N; i++)
            all_particles[i] = {all_x[i], all_y[i], all_vx[i], all_vy[i], all_mass[i]};
        writeCSV(all_particles, 0, csvFile);
    }

    // ---- Partial force accumulators (raw arrays for OMP reduction) ----
    // Size N: each process writes contributions for all particles whose
    // pair it is responsible for; zeros elsewhere.
    double* partial_fx = new double[N];
    double* partial_fy = new double[N];
    std::vector<double> total_fx(N), total_fy(N);

    // ---- Simulation loop ----
    double t_start = MPI_Wtime();

    for (int iter = 1; iter <= iterations; iter++) {

        // Share all current positions (velocities stay local)
        MPI_Allgatherv(local_x.data(), local_n, MPI_DOUBLE,
                       all_x.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_y.data(), local_n, MPI_DOUBLE,
                       all_y.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Reset partial accumulators
        for (int k = 0; k < N; k++) {
            partial_fx[k] = 0.0;
            partial_fy[k] = 0.0;
        }

        // --- OpenMP-parallel force computation with Newton's 3rd law ---
        // Each thread processes a subset of local particles (outer loop).
        // For each local particle i, only j > global_i pairs are computed;
        // the opposite force is written into partial_fx[j] / partial_fy[j].
        // OpenMP array reduction eliminates race conditions on the shared
        // partial arrays across threads within this process.
        #pragma omp parallel for schedule(dynamic, 16) \
                reduction(+: partial_fx[0:N], partial_fy[0:N])
        for (int i = 0; i < local_n; i++) {
            int global_i = local_start + i;
            for (int j = global_i + 1; j < N; j++) {
                double dx = all_x[j] - local_x[i];
                double dy = all_y[j] - local_y[i];

                double distSq   = dx * dx + dy * dy + SOFTENING * SOFTENING;
                double dist     = std::sqrt(distSq);
                double invDist3 = 1.0 / (distSq * dist);

                double F   = G * all_mass[global_i] * all_mass[j] * invDist3;
                double f_x = F * dx;
                double f_y = F * dy;

                partial_fx[global_i] += f_x;
                partial_fy[global_i] += f_y;
                partial_fx[j]        -= f_x;   // Newton's 3rd law
                partial_fy[j]        -= f_y;
            }
        }

        // Sum partial forces from all MPI processes
        MPI_Allreduce(partial_fx, total_fx.data(), N,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(partial_fy, total_fy.data(), N,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update velocities and positions (Euler integration)
        for (int i = 0; i < local_n; i++) {
            int global_i = local_start + i;
            double ax = total_fx[global_i] / all_mass[global_i];
            double ay = total_fy[global_i] / all_mass[global_i];

            local_vx[i] += ax * dt;
            local_vy[i] += ay * dt;

            local_x[i] += local_vx[i] * dt;
            local_y[i] += local_vy[i] * dt;
        }

        // Optionally gather full state on rank 0 and write CSV
        if (output_freq > 0 && iter % output_freq == 0) {
            MPI_Gatherv(local_x.data(),  local_n, MPI_DOUBLE,
                        all_x.data(),   counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_y.data(),  local_n, MPI_DOUBLE,
                        all_y.data(),   counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_vx.data(), local_n, MPI_DOUBLE,
                        all_vx.data(),  counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(local_vy.data(), local_n, MPI_DOUBLE,
                        all_vy.data(),  counts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::vector<Particle> all_particles(N);
                for (int i = 0; i < N; i++)
                    all_particles[i] = {all_x[i], all_y[i],
                                        all_vx[i], all_vy[i], all_mass[i]};
                writeCSV(all_particles, iter, csvFile);
            }
        }
    }

    double t_end  = MPI_Wtime();
    double elapsed = t_end - t_start;

    // ---- Results (rank 0 only) ----
    if (rank == 0) {
        std::cout << "Simulation complete.\n";
        std::cout << "Wall-clock time : " << elapsed << " seconds\n";
        std::cout << "Pairs/sec       : "
                  << (double)N * (N - 1) / 2.0 * iterations / elapsed << "\n";
        std::cout << "(Newton 3rd law + MPI_Allreduce + OpenMP array reduction)\n";
    }

    delete[] partial_fx;
    delete[] partial_fy;

    MPI_Finalize();
    return 0;
}
