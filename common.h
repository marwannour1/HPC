#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

// ============================================================
// Constants
// ============================================================
const double G = 6.674e-11;        // Gravitational constant
const double SOFTENING = 1e-3;     // Softening factor to avoid singularity

// ============================================================
// Particle structure (2D)
// ============================================================
struct Particle {
    double x, y;       // Position
    double vx, vy;     // Velocity
    double mass;        // Mass
};

// ============================================================
// Simulation parameters
// ============================================================
struct SimParams {
    int N;              // Number of particles
    double dt;          // Time step
    int iterations;     // Number of iterations
    int output_freq;    // Output CSV every N steps (0 = disable)
};

// ============================================================
// Parse command-line arguments
//   Usage: ./program <N> <dt> <iterations> [output_freq]
// ============================================================
inline SimParams parseArgs(int argc, char** argv) {
    SimParams params;
    params.N          = (argc > 1) ? std::atoi(argv[1]) : 500;
    params.dt         = (argc > 2) ? std::atof(argv[2]) : 0.01;
    params.iterations = (argc > 3) ? std::atoi(argv[3]) : 100;
    params.output_freq= (argc > 4) ? std::atoi(argv[4]) : 0;
    return params;
}

// ============================================================
// Initialize particles with a fixed random seed for
// reproducibility across all three implementations.
// Positions in [0, 1000], small random velocities,
// masses in [1e10, 1e12].
// ============================================================
inline void initParticles(int N, std::vector<Particle>& particles) {
    particles.resize(N);
    std::srand(42);  // Fixed seed for reproducibility

    for (int i = 0; i < N; i++) {
        particles[i].x    = ((double)std::rand() / RAND_MAX) * 1000.0;
        particles[i].y    = ((double)std::rand() / RAND_MAX) * 1000.0;
        particles[i].vx   = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
        particles[i].vy   = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        particles[i].mass = 1e10 + ((double)std::rand() / RAND_MAX) * (1e12 - 1e10);
    }
}

// ============================================================
// Write particle states to a CSV file.
// Format: step, id, x, y, vx, vy
// If step == 0, write the header first.
// ============================================================
inline void writeCSV(const std::vector<Particle>& particles, int step,
                     const std::string& filename)
{
    std::ofstream file;

    // On step 0 create the file with header; otherwise append
    if (step == 0) {
        file.open(filename, std::ios::out | std::ios::trunc);
        file << "step,id,x,y,vx,vy\n";
    } else {
        file.open(filename, std::ios::out | std::ios::app);
    }

    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return;
    }

    for (int i = 0; i < (int)particles.size(); i++) {
        file << step << "," << i << ","
             << particles[i].x  << "," << particles[i].y  << ","
             << particles[i].vx << "," << particles[i].vy << "\n";
    }

    file.close();
}

#endif // COMMON_H
