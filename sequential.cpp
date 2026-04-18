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

        // Compute pairwise gravitational forces
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i == j)
                    continue;

                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;

                // Distance with softening to prevent singularity
                double distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                double dist = std::sqrt(distSq);
                double invDist3 = 1.0 / (distSq * dist);

                // Gravitational force magnitude along each axis
                double F = G * particles[i].mass * particles[j].mass * invDist3;
                fx[i] += F * dx;
                fy[i] += F * dy;
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
    std::cout << "Interactions/sec: "
              << (double)N * N * iterations / elapsed.count() << "\n";

    return 0;
}
