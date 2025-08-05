import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_benchmark_full.py <benchmark_logfile>")
    sys.exit(1)

logfile = sys.argv[1]

# Regex für alle Werte
pattern = re.compile(
    r"Avg_Kernel_time=([\d\.]+)([ums]?)\s+"
    r"Avg_Wall_time=([\d\.]+)([ums]?).*?"
    r"Data_prep_time=([\d\.]+)([ums]?).*?"
    r"Flops_kerneltime=([\d\.]+)([TGMk]?).*?"
    r"Flops_walltime=([\d\.]+)([TGMk]?).*?"
    r"Kerneltime to walltime=([\d\.]+).*?"
    r"Sim_time=(\S+)"
)

unit_to_sec = {"": 1.0, "s": 1.0, "m": 1e-3, "u": 1e-6}
unit_to_flops = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

sim_times = []
kernel_times = []
wall_times = []
data_prep_times = []
flops_kernel = []
flops_wall = []

with open(logfile) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            (
                k_val, k_unit,
                w_val, w_unit,
                dp_val, dp_unit,
                fk_val, fk_unit,
                fw_val, fw_unit,
                ratio_val,
                sim_time_str
            ) = match.groups()

            kernel_s = float(k_val) * unit_to_sec[k_unit]
            wall_s = float(w_val) * unit_to_sec[w_unit]
            data_prep_s = float(dp_val) * unit_to_sec[dp_unit]

            # Umrechnung von FLOPS zu GFLOPS (sofern nötig)
            fk_raw = float(fk_val)
            fw_raw = float(fw_val)

            if fk_unit != "G":
                fk = fk_raw * unit_to_flops.get(fk_unit, 1.0)  # in FLOPS
                fk_gflops = fk / 1e9
            else:
                fk_gflops = fk_raw  # schon in GigaFLOPS

            if fw_unit != "G":
                fw = fw_raw * unit_to_flops.get(fw_unit, 1.0)
                fw_gflops = fw / 1e9
            else:
                fw_gflops = fw_raw

            # Sim_time in Zahl umwandeln
            if sim_time_str.endswith("k"):
                sim_time = float(sim_time_str[:-1]) * 1000
            else:
                sim_time = float(sim_time_str)

            sim_times.append(sim_time)
            kernel_times.append(kernel_s)
            wall_times.append(wall_s)
            data_prep_times.append(data_prep_s)
            flops_kernel.append(fk_gflops)
            flops_wall.append(fw_gflops)

# --- Plot 1: Kernel vs Wall Time ---
plt.figure(figsize=(8, 5))
plt.plot(sim_times, kernel_times, 'o-', label="Avg Kernel Time [s]")
plt.plot(sim_times, wall_times, 'o-', label="Avg Wall Time [s]")
plt.xscale('log')
plt.yscale('linear')
plt.xlabel("Sim_time")
plt.ylabel("Time [s]")
plt.title("Avg Kernel Time vs Avg Wall Time")
plt.xticks(sim_times, [str(int(x)) for x in sim_times], rotation=45)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{logfile}_kernel_vs_wall.png", dpi=300)

# --- Plot 2: FLOPS Kernel vs FLOPS Wall ---
plt.figure(figsize=(8, 5))
plt.plot(sim_times, flops_kernel, 'o-', label="Kernel FLOPS [GFLOP/s]")
plt.plot(sim_times, flops_wall, 'o-', label="Wall FLOPS [GFLOP/s]")
plt.xscale('log')
plt.yscale('linear')
plt.xlabel("Sim_time")
plt.ylabel("Performance [GFLOP/s]")
plt.title("FLOPS Kernel vs FLOPS Wall")
plt.xticks(sim_times, [str(int(x)) for x in sim_times], rotation=45)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{logfile}_flops_linear_gflops.png", dpi=300)

# --- Plot 3: Data Prep Time vs Kernel & Wall ---
plt.figure(figsize=(8, 5))
plt.plot(sim_times, data_prep_times, 'o-', label="Data Prep Time [s]")
plt.plot(sim_times, kernel_times, 'o-', label="Avg Kernel Time [s]")
plt.plot(sim_times, wall_times, 'o-', label="Avg Wall Time [s]")
plt.xscale('log')
plt.yscale('linear')
plt.xlabel("Sim_time")
plt.ylabel("Time [s]")
plt.title("Data Prep Time vs Kernel & Wall Time")
plt.xticks(sim_times, [str(int(x)) for x in sim_times], rotation=45)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{logfile}_dataprep_vs_times.png", dpi=300)

print("Plots gespeichert als PNG im aktuellen Ordner.")
