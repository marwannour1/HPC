// ============================================================
// Sequential N-Body Simulation (2D Gravitational)
// ============================================================
// Computes pairwise gravitational forces for all particles
// using a straightforward O(N^2) double loop.
// Positions and velocities are updated via Euler integration.
// ============================================================

#include "common.h"
#include <chrono>
#include <vector>

int main(int argc, char **argv)
{
    // ---- Parse arguments ----
    SimParams params = parseArgs(argc, argv);
    int N = params.N;
    double dt = params.dt;
    int iterations = params.iterations;
    int output_freq = params.output_freq;

    std::cout << "=== Sequential N-Body Simulation ===\n";
    std::cout << "Particles : " << N << "\n";
    std::cout << "Timestep  : " << dt << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    // ---- Initialize particles ----
    std::vector<Particle> particles;
    initParticles(N, particles);

    // Write initial state
    std::string csvFile = "output_sequential.csv";
    if (output_freq > 0)
    {
        writeCSV(particles, 0, csvFile);
    }

    // Force accumulators
    std::vector<double> fx(N), fy(N);

    // ---- Simulation loop ----
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= iterations; iter++)
    {

        // Reset forces
        for (int i = 0; i < N; i++)
        {
            fx[i] = 0.0;
            fy[i] = 0.0;
        }

        // Compute pairwise gravitational forces using Newton's 3rd law.
        // Only pairs (i, j) with j > i are evaluated — the force on j
        // is equal and opposite to the force on i, halving the work.
        for (int i = 0; i < N - 1; i++)
        {
            for (int j = i + 1; j < N; j++)
            {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;

                // Distance with softening to prevent singularity
                double distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                double dist = std::sqrt(distSq);
                double invDist3 = 1.0 / (distSq * dist);

                // Gravitational force magnitude along each axis
                double F = G * particles[i].mass * particles[j].mass * invDist3;
                double f_x = F * dx;
                double f_y = F * dy;

                fx[i] += f_x;
                fy[i] += f_y;
                fx[j] -= f_x;   // Newton's 3rd law: equal and opposite
                fy[j] -= f_y;
            }
        }

        // Update velocities and positions (Euler integration)
        for (int i = 0; i < N; i++)
        {
            double ax = fx[i] / particles[i].mass;
            double ay = fy[i] / particles[i].mass;

            particles[i].vx += ax * dt;
            particles[i].vy += ay * dt;

            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
        }

        // Optionally write CSV output
        if (output_freq > 0 && iter % output_freq == 0)
        {
            writeCSV(particles, iter, csvFile);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // ---- Results ----
    std::cout << "Simulation complete.\n";
    std::cout << "Wall-clock time : " << elapsed.count() << " seconds\n";
    std::cout << "Pairs/sec       : "
              << (double)N * (N - 1) / 2.0 * iterations / elapsed.count() << "\n";
    std::cout << "(Newton 3rd law: ~50% fewer force evals than naive)\n";

    return 0;
}
