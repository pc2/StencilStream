#!/usr/bin/env python3
import re
import sys
import json

# Constants
N_SUBITERATIONS = 3
OPERATIONS_PER_CELL = (
    (5 + 5 + 3 + 6 + 6 + 6) + (10 + 3 + 2 + 14 + 3 + 2) + 2
)
CELL_SIZE = 88  # bytes
LX = 768 * 1.5  # Modify as needed
LY = 768 * 0.5
RES = 1

# Open and load the JSON file
with open("/scratch/hpc-lco-kenter/tstoehr/sycl-stencil/examples/convection/experiments/default.json", "r") as f:
    data = json.load(f)

# Extract values
ly = data["ly"]
lx = data["lx"]
res = data["res"]

# Compute grid dimensions
grid_height = int(ly * res)
grid_width = int(lx * res)

# Print results
print(f"Grid height: {grid_height}")
print(f"Grid width: {grid_width}")

def analyze_slurm_log(filepath):
    # Compile regex patterns
    iteration_re = re.compile(r"it = \d+ \(iter = (\d+), time = ([\d.eE+-]+)\)")
    #transient_time_re = re.compile(r"Of which transient computation time: ([\d.eE+-]+) s$")
    transient_time_re = re.compile(r"Of which transient computation kerneltime: ([\d.eE+-]+) s$")

    total_iterations = 0
    transient_time = 0.0

    with open(filepath, "r") as f:
        for line in f:
            match = iteration_re.search(line)
            match2 = transient_time_re.search(line)

            if match:
                iter_count = int(match.group(1))
                total_iterations += iter_count

            if match2:
                transient_time = float(match2.group(1))

    return total_iterations, transient_time


def calculate_metrics(iterations, time):
    total_cells = grid_width * grid_height
    workload = iterations * total_cells
    throughput = workload / time
    flops = throughput * OPERATIONS_PER_CELL
    gflops = flops / 1e9
    return iterations, time, throughput, gflops

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_runtime.py <slurm_log.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    iterations, total_time = analyze_slurm_log(filepath)
    iterations, total_time, throughput, gflops = calculate_metrics(iterations, total_time)

    print(f"Iterations: {iterations}")
    print(f"Total Time: {total_time:.6f} s")
    print(f"Throughput: {throughput:.2f} cells/s")
    print(f"GFLOP/s: {gflops:.2f}")
