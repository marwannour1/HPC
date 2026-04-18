// ============================================================
// OpenMP Parallel N-Body Simulation (2D Gravitational)
// ============================================================
// Uses Newton's 3rd law to compute each pair (i, j) only once,
// halving the work compared to the sequential version.
// Because multiple threads may update the same force array
// elements, we use OpenMP array reduction to avoid race
// conditions on the shared force accumulators.
// ============================================================

#include "common.h"
#include <chrono>
#include <vector>
#include <omp.h>

int main(int argc, char** argv) {
    // ---- Parse arguments ----
    SimParams params = parseArgs(argc, argv);
    int N           = params.N;
    double dt       = params.dt;
    int iterations  = params.iterations;
    int output_freq = params.output_freq;

    int num_threads = omp_get_max_threads();

    std::cout << "=== OpenMP N-Body Simulation ===\n";
    std::cout << "Particles : " << N << "\n";
    std::cout << "Timestep  : " << dt << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Threads   : " << num_threads << "\n\n";

    // ---- Initialize particles ----
    std::vector<Particle> particles;
    initParticles(N, particles);

    // Write initial state
    std::string csvFile = "output_openmp.csv";
    if (output_freq > 0) {
        writeCSV(particles, 0, csvFile);
    }

    // Force accumulators — raw pointers for OpenMP array reduction
    double* fx = new double[N];
    double* fy = new double[N];

    // ---- Simulation loop ----
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= iterations; iter++) {

        // Reset forces
        for (int i = 0; i < N; i++) {
            fx[i] = 0.0;
            fy[i] = 0.0;
        }

        // Compute pairwise gravitational forces using Newton's 3rd law.
        // Each pair (i, j) with j > i is computed once — the force on j
        // is the negative of the force on i, halving the total work.
        // Multiple threads may update the same fx[j]/fy[j], so we use
        // OpenMP array reduction to eliminate race conditions.
        #pragma omp parallel for schedule(dynamic, 16) reduction(+:fx[:N], fy[:N])
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;

                double distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                double dist   = std::sqrt(distSq);
                double invDist3 = 1.0 / (distSq * dist);

                double F = G * particles[i].mass * particles[j].mass * invDist3;
                double f_x = F * dx;
                double f_y = F * dy;

                fx[i] += f_x;
                fy[i] += f_y;
                fx[j] -= f_x;   // Newton's 3rd law
                fy[j] -= f_y;
            }
        }

        // Update velocities and positions (Euler integration)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double ax = fx[i] / particles[i].mass;
            double ay = fy[i] / particles[i].mass;

            particles[i].vx += ax * dt;
            particles[i].vy += ay * dt;

            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
        }

        // Optionally write CSV output
        if (output_freq > 0 && iter % output_freq == 0) {
            writeCSV(particles, iter, csvFile);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ---- Results ----
    std::cout << "Simulation complete.\n";
    std::cout << "Wall-clock time : " << elapsed.count() << " seconds\n";
    std::cout << "Interactions/sec: "
              << (double)N * N * iterations / elapsed.count() << "\n";

    delete[] fx;
    delete[] fy;

    return 0;
}
