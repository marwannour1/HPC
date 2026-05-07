# ============================================================
# benchmark.ps1  -  Automated N-Body Performance Benchmark
# ============================================================
# Compiles all four versions (sequential, openmp, mpi, hybrid),
# runs them across increasing particle counts, then writes
# timing, speedup, and efficiency results to benchmark_results.csv
#
# Usage:
#   .\benchmark.ps1
#   .\benchmark.ps1 -SkipBuild        # skip compilation
#   .\benchmark.ps1 -MaxN 2000        # override max particle count
# ============================================================

param(
    [switch]$SkipBuild,
    [int]$MaxN = 5000
)

Set-Location $PSScriptRoot

# ---- Build paths ----
$MPI_INC = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
$MPI_LIB = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
$CXX     = "g++"
$FLAGS   = @("-O2", "-std=c++17", "-Wall")

# ---- Experiment parameters ----
# Particle counts to sweep (filtered by MaxN)
$AllN        = @(100, 500, 1000, 2000, 5000) | Where-Object { $_ -le $MaxN }
$OmpThreads  = @(2, 4, 8)      # OMP_NUM_THREADS values to test
$MpiProcs    = @(2, 4)         # mpiexec -n values to test
# Hybrid combos: (processes x threads) — kept small to avoid explosion
$HybridCombos = @(
    @{P=2; T=2},
    @{P=2; T=4},
    @{P=4; T=2}
)

# Iterations scale down for large N to keep wall time reasonable
function Get-Iterations([int]$N) {
    if ($N -le  500) { return 100 }
    if ($N -le 1000) { return  50 }
    if ($N -le 2000) { return  30 }
    return 15
}

# ---- Helper: extract wall-clock time from program output ----
function Get-WallTime([string[]]$output) {
    $line = $output | Where-Object { $_ -match "Wall-clock time" } | Select-Object -First 1
    if ($line -match "([\d.]+(?:e[+\-]?\d+)?)\s*seconds") {
        return [double]$Matches[1]
    }
    return $null
}

# ============================================================
# Build
# ============================================================
if (-not $SkipBuild) {
    Write-Host "`n=== Compiling all versions ===" -ForegroundColor Cyan

    Write-Host "  Building sequential.exe ..."
    & $CXX @FLAGS -o sequential.exe sequential.cpp
    if ($LASTEXITCODE -ne 0) { Write-Error "sequential build FAILED"; exit 1 }

    Write-Host "  Building openmp.exe ..."
    & $CXX @FLAGS -fopenmp -o openmp.exe openmp.cpp
    if ($LASTEXITCODE -ne 0) { Write-Error "openmp build FAILED"; exit 1 }

    Write-Host "  Building mpi.exe ..."
    & $CXX @FLAGS -o mpi.exe mpi.cpp -I $MPI_INC -L $MPI_LIB -lmsmpi
    if ($LASTEXITCODE -ne 0) { Write-Error "mpi build FAILED"; exit 1 }

    Write-Host "  Building hybrid.exe ..."
    & $CXX @FLAGS -fopenmp -o hybrid.exe hybrid.cpp -I $MPI_INC -L $MPI_LIB -lmsmpi
    if ($LASTEXITCODE -ne 0) { Write-Error "hybrid build FAILED"; exit 1 }

    Write-Host "  All builds successful.`n" -ForegroundColor Green
}

# ============================================================
# Run experiments
# ============================================================
$Results = [System.Collections.Generic.List[PSCustomObject]]::new()
$dt = 0.01

foreach ($N in $AllN) {
    $iters = Get-Iterations $N
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  N = $N   (iterations = $iters)" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    # ------ Sequential baseline ------
    Write-Host "  [Sequential]" -NoNewline
    $out     = & .\sequential.exe $N $dt $iters 2>&1
    $seqTime = Get-WallTime $out
    Write-Host "  $seqTime s"

    $Results.Add([PSCustomObject]@{
        N          = $N
        Iterations = $iters
        Version    = "Sequential"
        Processes  = 1
        Threads    = 1
        Workers    = 1
        Time_s     = $seqTime
        Speedup    = 1.0
        Efficiency = 1.0
    })

    # ------ OpenMP ------
    foreach ($t in $OmpThreads) {
        $env:OMP_NUM_THREADS = "$t"
        Write-Host "  [OpenMP  T=$t]" -NoNewline
        $out     = & .\openmp.exe $N $dt $iters 2>&1
        $ompTime = Get-WallTime $out
        $speedup = if ($ompTime -and $seqTime) { [math]::Round($seqTime / $ompTime, 4) } else { $null }
        $eff     = if ($speedup)               { [math]::Round($speedup / $t, 4)        } else { $null }
        Write-Host "  $ompTime s  |  speedup=$speedup  efficiency=$eff"

        $Results.Add([PSCustomObject]@{
            N          = $N
            Iterations = $iters
            Version    = "OpenMP"
            Processes  = 1
            Threads    = $t
            Workers    = $t
            Time_s     = $ompTime
            Speedup    = $speedup
            Efficiency = $eff
        })
    }

    # ------ MPI ------
    foreach ($p in $MpiProcs) {
        Write-Host "  [MPI     P=$p]" -NoNewline
        $out     = & mpiexec -n $p .\mpi.exe $N $dt $iters 2>&1
        $mpiTime = Get-WallTime $out
        $speedup = if ($mpiTime -and $seqTime) { [math]::Round($seqTime / $mpiTime, 4) } else { $null }
        $eff     = if ($speedup)               { [math]::Round($speedup / $p, 4)        } else { $null }
        Write-Host "  $mpiTime s  |  speedup=$speedup  efficiency=$eff"

        $Results.Add([PSCustomObject]@{
            N          = $N
            Iterations = $iters
            Version    = "MPI"
            Processes  = $p
            Threads    = 1
            Workers    = $p
            Time_s     = $mpiTime
            Speedup    = $speedup
            Efficiency = $eff
        })
    }

    # ------ Hybrid ------
    foreach ($combo in $HybridCombos) {
        $p = $combo.P
        $t = $combo.T
        $env:OMP_NUM_THREADS = "$t"
        Write-Host "  [Hybrid  P=$p x T=$t]" -NoNewline
        $out      = & mpiexec -n $p .\hybrid.exe $N $dt $iters 2>&1
        $hybTime  = Get-WallTime $out
        $workers  = $p * $t
        $speedup  = if ($hybTime -and $seqTime) { [math]::Round($seqTime / $hybTime, 4) } else { $null }
        $eff      = if ($speedup)               { [math]::Round($speedup / $workers, 4)  } else { $null }
        Write-Host "  $hybTime s  |  speedup=$speedup  efficiency=$eff  workers=$workers"

        $Results.Add([PSCustomObject]@{
            N          = $N
            Iterations = $iters
            Version    = "Hybrid"
            Processes  = $p
            Threads    = $t
            Workers    = $workers
            Time_s     = $hybTime
            Speedup    = $speedup
            Efficiency = $eff
        })
    }
}

# ============================================================
# Write CSV
# ============================================================
$CsvPath = Join-Path $PSScriptRoot "benchmark_results.csv"
$Results | Export-Csv -Path $CsvPath -NoTypeInformation

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Results saved to: $CsvPath"
Write-Host "Rows written    : $($Results.Count)"
