#!/usr/bin/env python3
"""
Performance benchmark for q2netcdf optimizations.

Measures:
1. Configuration parsing speed (regex caching)
2. Data record parsing speed (array pre-allocation)
3. File I/O throughput (buffering)
"""

import time
import numpy as np
from pathlib import Path
from q2netcdf import QFile


def benchmark_qfile_reading(qfile_path: str, iterations: int = 3) -> dict:
    """Benchmark complete Q-file reading performance."""
    times = []
    records_read = 0

    for i in range(iterations):
        start = time.perf_counter()
        with QFile(qfile_path) as qf:
            hdr = qf.header()
            count = 0
            for record in qf.data():
                count += 1
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        records_read = count

    return {
        "file": Path(qfile_path).name,
        "records": records_read,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "records_per_sec": records_read / np.mean(times),
    }


def benchmark_config_parsing(qfile_path: str, iterations: int = 10) -> dict:
    """Benchmark configuration parsing (regex caching impact)."""
    times = []

    for i in range(iterations):
        with QFile(qfile_path) as qf:
            start = time.perf_counter()
            hdr = qf.header()
            config = hdr.config.config()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        "operation": "config_parsing",
        "iterations": iterations,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def benchmark_header_reading(qfile_path: str, iterations: int = 100) -> dict:
    """Benchmark header reading performance."""
    times = []

    for i in range(iterations):
        with QFile(qfile_path) as qf:
            start = time.perf_counter()
            hdr = qf.header()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        "operation": "header_reading",
        "iterations": iterations,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def format_results(results: dict, title: str) -> str:
    """Format benchmark results for display."""
    lines = [f"\n{'='*60}", f"{title:^60}", f"{'='*60}"]

    if "file" in results:
        lines.append(f"File:              {results['file']}")
        lines.append(f"Records:           {results['records']}")
        lines.append(f"Mean time:         {results['mean_time']:.4f} s")
        lines.append(f"Std dev:           {results['std_time']:.4f} s")
        lines.append(f"Min time:          {results['min_time']:.4f} s")
        lines.append(f"Max time:          {results['max_time']:.4f} s")
        lines.append(f"Throughput:        {results['records_per_sec']:.1f} records/s")
    else:
        lines.append(f"Operation:         {results['operation']}")
        lines.append(f"Iterations:        {results['iterations']}")
        lines.append(f"Mean time:         {results['mean_time']*1000:.2f} ms")
        lines.append(f"Std dev:           {results['std_time']*1000:.2f} ms")
        lines.append(f"Min time:          {results['min_time']*1000:.2f} ms")
        lines.append(f"Max time:          {results['max_time']*1000:.2f} ms")

    lines.append("="*60)
    return "\n".join(lines)


def main():
    """Run all benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark q2netcdf performance optimizations"
    )
    parser.add_argument(
        "qfile",
        type=str,
        help="Path to Q-file for benchmarking",
    )
    parser.add_argument(
        "--full-iterations",
        type=int,
        default=3,
        help="Number of full file read iterations (default: 3)",
    )
    parser.add_argument(
        "--config-iterations",
        type=int,
        default=10,
        help="Number of config parsing iterations (default: 10)",
    )
    parser.add_argument(
        "--header-iterations",
        type=int,
        default=100,
        help="Number of header reading iterations (default: 100)",
    )

    args = parser.parse_args()

    print("\nq2netcdf Performance Benchmark")
    print("==============================")
    print(f"\nOptimizations tested:")
    print("  1. Regex pattern caching in QConfig")
    print("  2. Numpy array pre-allocation in QRecord")
    print("  3. 64KB buffered I/O in QFile")

    # Full file reading benchmark
    print("\n[1/3] Full file reading benchmark...")
    results = benchmark_qfile_reading(args.qfile, args.full_iterations)
    print(format_results(results, "Full File Reading"))

    # Configuration parsing benchmark
    print("\n[2/3] Configuration parsing benchmark...")
    results = benchmark_config_parsing(args.qfile, args.config_iterations)
    print(format_results(results, "Configuration Parsing"))

    # Header reading benchmark
    print("\n[3/3] Header reading benchmark...")
    results = benchmark_header_reading(args.qfile, args.header_iterations)
    print(format_results(results, "Header Reading"))

    print("\n✓ Benchmark complete!\n")


if __name__ == "__main__":
    main()
