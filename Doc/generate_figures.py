from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "benchmark_results.csv"
LOGO_SOURCE = ROOT / "Ain-Shams-University-Egypt-34749-1533823446.webp"
LOGO_TARGET = ROOT / "uni_logo.png"


def load_rows() -> list[dict[str, object]]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            rows.append(
                {
                    "N": int(raw["N"]),
                    "Iterations": int(raw["Iterations"]),
                    "Version": raw["Version"],
                    "Processes": int(raw["Processes"]),
                    "Threads": int(raw["Threads"]),
                    "Workers": int(raw["Workers"]),
                    "Time_s": float(raw["Time_s"]),
                    "Speedup": float(raw["Speedup"]),
                    "Efficiency": float(raw["Efficiency"]),
                }
            )
    return rows


def select_series(
    rows: list[dict[str, object]],
    *,
    version: str,
    processes: int | None = None,
    threads: int | None = None,
    workers: int | None = None,
) -> list[dict[str, object]]:
    series = []
    for row in rows:
        if row["Version"] != version:
            continue
        if processes is not None and row["Processes"] != processes:
            continue
        if threads is not None and row["Threads"] != threads:
            continue
        if workers is not None and row["Workers"] != workers:
            continue
        series.append(row)
    return sorted(series, key=lambda row: row["N"])


def pairs_per_second(row: dict[str, object]) -> float:
    n = int(row["N"])
    iterations = int(row["Iterations"])
    time_s = float(row["Time_s"])
    return (n * (n - 1) / 2.0) * iterations / time_s


def convert_logo() -> None:
    if not LOGO_SOURCE.exists():
        return
    image = Image.open(LOGO_SOURCE)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    image.save(LOGO_TARGET)


def setup_plot() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_walltime(rows: list[dict[str, object]]) -> None:
    configs = [
        ("Seq", select_series(rows, version="Sequential"), "#111827", "o"),
        ("OMP 2T", select_series(rows, version="OpenMP", threads=2), "#2563eb", "o"),
        ("OMP 4T", select_series(rows, version="OpenMP", threads=4), "#1d4ed8", "s"),
        ("OMP 8T", select_series(rows, version="OpenMP", threads=8), "#0f172a", "^"),
        ("MPI 2P", select_series(rows, version="MPI", processes=2), "#059669", "D"),
        ("MPI 4P", select_series(rows, version="MPI", processes=4), "#047857", "v"),
        ("Hyb 2x4", select_series(rows, version="Hybrid", processes=2, threads=4), "#dc2626", "P"),
        ("Hyb 4x2", select_series(rows, version="Hybrid", processes=4, threads=2), "#b91c1c", "X"),
    ]

    plt.figure(figsize=(10.5, 6.6))
    for label, series, color, marker in configs:
        x = [int(row["N"]) for row in series]
        y = [float(row["Time_s"]) for row in series]
        plt.plot(x, y, marker=marker, linewidth=2.2, markersize=7, label=label, color=color)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Particle count N")
    plt.ylabel("Wall-clock time (s)")
    plt.title("Wall-Clock Time vs Particle Count")
    plt.legend(ncol=2)
    save_figure(ROOT / "fig1_walltime.png")


def plot_openmp_speedup(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10.2, 6.2))
    openmp_series = [
        (2, "#2563eb", "o"),
        (4, "#1d4ed8", "s"),
        (8, "#0f172a", "^"),
    ]
    for threads, color, marker in openmp_series:
        series = select_series(rows, version="OpenMP", threads=threads)
        plt.plot(
            [int(row["N"]) for row in series],
            [float(row["Speedup"]) for row in series],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            color=color,
            label=f"OMP {threads}T",
        )
        plt.axhline(threads, linestyle="--", linewidth=1.2, color=color, alpha=0.28)
    plt.xlabel("Particle count N")
    plt.ylabel("Speedup")
    plt.title("OpenMP Speedup vs Particle Count")
    plt.legend()
    save_figure(ROOT / "fig2_speedup_openmp.png")


def plot_mpi_speedup(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10.2, 6.2))
    mpi_series = [
        (2, "#059669", "D"),
        (4, "#047857", "v"),
    ]
    for processes, color, marker in mpi_series:
        series = select_series(rows, version="MPI", processes=processes)
        plt.plot(
            [int(row["N"]) for row in series],
            [float(row["Speedup"]) for row in series],
            marker=marker,
            linewidth=2.4,
            markersize=7,
            color=color,
            label=f"MPI {processes}P",
        )
    plt.xlabel("Particle count N")
    plt.ylabel("Speedup")
    plt.title("MPI Speedup vs Particle Count")
    plt.legend()
    save_figure(ROOT / "fig3_speedup_mpi.png")


def plot_hybrid_speedup(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10.2, 6.2))
    series_defs = [
        ("Hybrid 2P x 4T", select_series(rows, version="Hybrid", processes=2, threads=4), "#dc2626", "P"),
        ("Hybrid 4P x 2T", select_series(rows, version="Hybrid", processes=4, threads=2), "#b91c1c", "X"),
        ("OpenMP 8T", select_series(rows, version="OpenMP", threads=8), "#0f172a", "^"),
    ]
    for label, series, color, marker in series_defs:
        plt.plot(
            [int(row["N"]) for row in series],
            [float(row["Speedup"]) for row in series],
            marker=marker,
            linewidth=2.4,
            markersize=7,
            color=color,
            label=label,
        )
    plt.xlabel("Particle count N")
    plt.ylabel("Speedup")
    plt.title("Hybrid vs OpenMP Speedup")
    plt.legend()
    save_figure(ROOT / "fig4_speedup_hybrid.png")


def plot_efficiency(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10.2, 6.2))
    series_defs = [
        ("OMP 8T", select_series(rows, version="OpenMP", threads=8), "#0f172a", "^", "-"),
        ("Hybrid 2P x 4T", select_series(rows, version="Hybrid", processes=2, threads=4), "#dc2626", "P", "-"),
        ("Hybrid 4P x 2T", select_series(rows, version="Hybrid", processes=4, threads=2), "#b91c1c", "X", "-"),
        ("MPI 4P", select_series(rows, version="MPI", processes=4), "#047857", "v", "--"),
    ]
    for label, series, color, marker, linestyle in series_defs:
        plt.plot(
            [int(row["N"]) for row in series],
            [float(row["Efficiency"]) for row in series],
            marker=marker,
            linewidth=2.4,
            markersize=7,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    plt.ylim(0, 1.05)
    plt.xlabel("Particle count N")
    plt.ylabel("Efficiency")
    plt.title("Parallel Efficiency Comparison")
    plt.legend()
    save_figure(ROOT / "fig5_efficiency.png")


def plot_overhead(rows: list[dict[str, object]]) -> None:
    n100_rows = [row for row in rows if int(row["N"]) == 100]
    seq_time_ms = next(float(row["Time_s"]) * 1000.0 for row in n100_rows if row["Version"] == "Sequential")
    series_defs = [
        ("Seq", next(row for row in n100_rows if row["Version"] == "Sequential"), "#111827"),
        ("OMP 2T", next(row for row in n100_rows if row["Version"] == "OpenMP" and row["Threads"] == 2), "#2563eb"),
        ("OMP 4T", next(row for row in n100_rows if row["Version"] == "OpenMP" and row["Threads"] == 4), "#1d4ed8"),
        ("OMP 8T", next(row for row in n100_rows if row["Version"] == "OpenMP" and row["Threads"] == 8), "#0f172a"),
        ("MPI 2P", next(row for row in n100_rows if row["Version"] == "MPI" and row["Processes"] == 2), "#059669"),
        ("MPI 4P", next(row for row in n100_rows if row["Version"] == "MPI" and row["Processes"] == 4), "#047857"),
        ("Hyb 2x4", next(row for row in n100_rows if row["Version"] == "Hybrid" and row["Processes"] == 2 and row["Threads"] == 4), "#dc2626"),
        ("Hyb 4x2", next(row for row in n100_rows if row["Version"] == "Hybrid" and row["Processes"] == 4 and row["Threads"] == 2), "#b91c1c"),
    ]

    labels = [label for label, _, _ in series_defs]
    values = [float(row["Time_s"]) * 1000.0 for _, row, _ in series_defs]
    colors = [color for _, _, color in series_defs]
    max_value = max(values)

    plt.figure(figsize=(11.5, 6.4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Wall-clock time (ms)")
    plt.title("Overhead-Dominated Regime at N=100", pad=14)
    plt.ylim(0, max_value * 1.18)
    plt.xticks(rotation=20)

    for bar, (_, row, _) in zip(bars, series_defs):
        time_ms = float(row["Time_s"]) * 1000.0
        ratio = time_ms / seq_time_ms
        text = "baseline" if row["Version"] == "Sequential" else f"{ratio:.2f}x seq"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.18,
            text,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9,
        )
    save_figure(ROOT / "fig6_overhead_n100.png")


def plot_pairs_per_second(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(10.2, 6.2))
    baseline = select_series(rows, version="Sequential")
    best_parallel: list[dict[str, object]] = []
    by_n: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_n.setdefault(int(row["N"]), []).append(row)
    for n, group in sorted(by_n.items()):
        parallel_group = [row for row in group if row["Version"] != "Sequential"]
        best_parallel.append(max(parallel_group, key=pairs_per_second))

    plt.plot(
        [int(row["N"]) for row in baseline],
        [pairs_per_second(row) / 1_000_000.0 for row in baseline],
        marker="o",
        linewidth=2.4,
        color="#111827",
        label="Sequential",
    )
    plt.plot(
        [int(row["N"]) for row in best_parallel],
        [pairs_per_second(row) / 1_000_000.0 for row in best_parallel],
        marker="^",
        linewidth=2.4,
        color="#dc2626",
        label="Best parallel config",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Particle count N")
    plt.ylabel("Pairs per second (millions)")
    plt.title("Computation Throughput")
    plt.legend()
    save_figure(ROOT / "fig_pairs_per_second.png")


def main() -> None:
    rows = load_rows()
    setup_plot()
    convert_logo()
    plot_walltime(rows)
    plot_openmp_speedup(rows)
    plot_mpi_speedup(rows)
    plot_hybrid_speedup(rows)
    plot_efficiency(rows)
    plot_overhead(rows)
    plot_pairs_per_second(rows)
    print("Generated:")
    for path in [
        LOGO_TARGET,
        ROOT / "fig1_walltime.png",
        ROOT / "fig2_speedup_openmp.png",
        ROOT / "fig3_speedup_mpi.png",
        ROOT / "fig4_speedup_hybrid.png",
        ROOT / "fig5_efficiency.png",
        ROOT / "fig6_overhead_n100.png",
        ROOT / "fig_pairs_per_second.png",
    ]:
        if path.exists():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()