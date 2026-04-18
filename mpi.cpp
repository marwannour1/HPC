// ============================================================
// MPI Parallel N-Body Simulation (2D Gravitational)
// ============================================================
// Distributes particles among MPI processes.  Each timestep,
// all processes exchange particle positions and masses via
// MPI_Allgatherv so that every process can compute forces for
// its local subset against the full particle set.
// ============================================================

#include "common.h"
#include <mpi.h>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parse arguments ----
    SimParams params = parseArgs(argc, argv);
    int N           = params.N;
    double dt       = params.dt;
    int iterations  = params.iterations;
    int output_freq = params.output_freq;

    if (rank == 0) {
        std::cout << "=== MPI N-Body Simulation ===\n";
        std::cout << "Particles : " << N << "\n";
        std::cout << "Timestep  : " << dt << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Processes : " << size << "\n\n";
    }

    // ---- Determine how particles are distributed ----
    // Each process gets N/size particles; the last process
    // absorbs the remainder.
    std::vector<int> counts(size), displs(size);
    int base = N / size;
    int remainder = N % size;
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }
    int local_n = counts[rank];
    int local_start = displs[rank];

    // ---- Initialize all particles on rank 0 and scatter ----
    // We store the full particle arrays (x, y, vx, vy, mass) as
    // separate flat arrays for easy MPI communication.
    std::vector<double> all_x(N), all_y(N);
    std::vector<double> all_vx(N), all_vy(N);
    std::vector<double> all_mass(N);

    if (rank == 0) {
        // Use the same initialization as sequential/openmp
        std::vector<Particle> temp;
        initParticles(N, temp);
        for (int i = 0; i < N; i++) {
            all_x[i]    = temp[i].x;
            all_y[i]    = temp[i].y;
            all_vx[i]   = temp[i].vx;
            all_vy[i]   = temp[i].vy;
            all_mass[i]  = temp[i].mass;
        }
    }

    // Broadcast masses (they don't change)
    MPI_Bcast(all_mass.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter initial positions and velocities to local arrays
    std::vector<double> local_x(local_n), local_y(local_n);
    std::vector<double> local_vx(local_n), local_vy(local_n);

    MPI_Scatterv(all_x.data(),  counts.data(), displs.data(), MPI_DOUBLE,
                 local_x.data(),  local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_y.data(),  counts.data(), displs.data(), MPI_DOUBLE,
                 local_y.data(),  local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vx.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local_vx.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_vy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local_vy.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Write initial state (rank 0 gathers first)
    std::string csvFile = "output_mpi.csv";
    if (output_freq > 0 && rank == 0) {
        // Reconstruct Particle vector for CSV writing
        std::vector<Particle> all_particles(N);
        // At this point rank 0 still has all_x/y/vx/vy from init
        for (int i = 0; i < N; i++) {
            all_particles[i] = {all_x[i], all_y[i], all_vx[i], all_vy[i], all_mass[i]};
        }
        writeCSV(all_particles, 0, csvFile);
    }

    // Force accumulators (local)
    std::vector<double> fx(local_n), fy(local_n);

    // ---- Simulation loop ----
    double t_start = MPI_Wtime();

    for (int iter = 1; iter <= iterations; iter++) {

        // Gather all positions so every process has the full set.
        // Velocities are not needed by other processes.
        MPI_Allgatherv(local_x.data(), local_n, MPI_DOUBLE,
                        all_x.data(),  counts.data(), displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_y.data(), local_n, MPI_DOUBLE,
                        all_y.data(),  counts.data(), displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute forces for local particles against all N particles
        for (int i = 0; i < local_n; i++) {
            fx[i] = 0.0;
            fy[i] = 0.0;

            int global_i = local_start + i;
            for (int j = 0; j < N; j++) {
                if (j == global_i) continue;

                double dx = all_x[j] - local_x[i];
                double dy = all_y[j] - local_y[i];

                double distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                double dist   = std::sqrt(distSq);
                double invDist3 = 1.0 / (distSq * dist);

                double F = G * all_mass[global_i] * all_mass[j] * invDist3;
                fx[i] += F * dx;
                fy[i] += F * dy;
            }
        }

        // Update velocities and positions for local particles
        for (int i = 0; i < local_n; i++) {
            int global_i = local_start + i;
            double ax = fx[i] / all_mass[global_i];
            double ay = fy[i] / all_mass[global_i];

            local_vx[i] += ax * dt;
            local_vy[i] += ay * dt;

            local_x[i] += local_vx[i] * dt;
            local_y[i] += local_vy[i] * dt;
        }

        // Optionally gather & write CSV
        if (output_freq > 0 && iter % output_freq == 0) {
            // Gather positions and velocities on rank 0
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
                for (int i = 0; i < N; i++) {
                    all_particles[i] = {all_x[i], all_y[i],
                                        all_vx[i], all_vy[i], all_mass[i]};
                }
                writeCSV(all_particles, iter, csvFile);
            }
        }
    }

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // ---- Results (rank 0 only) ----
    if (rank == 0) {
        std::cout << "Simulation complete.\n";
        std::cout << "Wall-clock time : " << elapsed << " seconds\n";
        std::cout << "Interactions/sec: "
                  << (double)N * N * iterations / elapsed << "\n";
    }

    MPI_Finalize();
    return 0;
}
