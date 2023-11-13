#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")

const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const N_MONOTILE_CUS = 50
const N_TILING_CUS = 50
const TILE_SIZE = 1024

if size(ARGS) != (2,)
    println(stderr, "Usage: $PROGRAM_FILE <path to executable> <variant>")
    exit(1)
end

exec = ARGS[1]
report_path = exec * ".prj/reports"
f, loop_latency = load_report_details(report_path)

variant = ARGS[2]
if variant == "monotile"
    n_cus = N_MONOTILE_CUS
    variant = :monotile
elseif variant == "tiling"
    n_cus = N_TILING_CUS
    variant = :tiling
else
    println(stderr, "Unknown variant '$variant'")
    exit(1)
end
n_gens = n_cus

command = `$exec $TILE_SIZE $TILE_SIZE $n_gens ./data/temp_$TILE_SIZE ./data/power_$TILE_SIZE /dev/null`

# Run the simulation once to eliminate the FPGA programming from the measured runtime
run(command)

open(command, "r") do process_in
    runtime = nothing
    while runtime === nothing
        line = readline(process_in)
        println(line)

        line_match = match(r"Kernel Runtime: ([0-9]+\.[0-9]+) s", line)
        if line_match !== nothing
            runtime = parse(Float64, line_match[1])
        end
    end

    metrics = build_metrics(
        (variant == :monotile) ? "Hotspot, Monotile" : "Hotspot, Tiling",
        runtime,
        n_gens,
        variant,
        f,
        loop_latency,
        1024,
        1024,
        TILE_SIZE,
        TILE_SIZE,
        n_cus,
        OPERATIONS_PER_CELL,
        CELL_SIZE
    )
    
    open("metrics.$variant.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end