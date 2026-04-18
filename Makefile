# ============================================================
# Makefile for N-Body Simulation Project
# ============================================================
# Targets:
#   make all        - Build all three versions
#   make sequential - Build sequential version
#   make openmp     - Build OpenMP version
#   make mpi        - Build MPI version
#   make clean      - Remove executables and CSV output
# ============================================================

CXX      = g++
MPICXX   = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall

all: sequential openmp mpi

sequential: sequential.cpp common.h
	$(CXX) $(CXXFLAGS) -o sequential sequential.cpp -lm

openmp: openmp.cpp common.h
	$(CXX) $(CXXFLAGS) -fopenmp -o openmp openmp.cpp -lm

mpi: mpi.cpp common.h
	$(MPICXX) $(CXXFLAGS) -o mpi mpi.cpp -lm

clean:
	rm -f sequential openmp mpi *.csv

.PHONY: all clean
