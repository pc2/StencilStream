#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")

const N_SUBGENERATIONS = 2
const N_TILING_CUS = 50
const N_MONOTILE_CUS = 50
const OPERATIONS_PER_CELL = 0.5 * (8) + 0.5 * (6+4+2+2+2) # Including all paths, excluding source wave computation
const CELL_SIZE = 4 * (4 + 4) # bytes, including material coefficients
const TILE_HEIGHT = 512
const TILING_TILE_WIDTH = 2^16
const MONOTILE_TILE_WIDTH = 512

if size(ARGS) != (2,)
    println(stderr, "Usage: $PROGRAM_FILE <path to executable> <variant>")
    exit(1)
end

exe = ARGS[1]
report_path = exe * ".prj/reports"
f, loop_latency = load_report_details(report_path)

variant = ARGS[2]
if variant == "monotile"
    n_cus = N_MONOTILE_CUS
    variant = :monotile
    tile_height, tile_width = TILE_HEIGHT, MONOTILE_TILE_WIDTH
elseif variant == "tiling"
    n_cus = N_TILING_CUS
    variant = :tiling
    tile_height, tile_width = TILE_HEIGHT, TILING_TILE_WIDTH
else
    println(stderr, "Unknown variant '$variant'")
    exit(1)
end

command = `$exe -c ./experiments/default.json`
# Run the simulation once to eliminate the FPGA programming from the measured runtime
run(command)

open(command, "r") do process_in
    runtime = nothing
    grid_wh = nothing
    n_timesteps = nothing

    # While any of the metrics is nothing...
    while any(v -> v === nothing, [runtime, grid_wh, n_timesteps])
        line = readline(process_in)
        println(line)
        if (m = match(r"grid w/h      = ([0-9]+) cells", line)) !== nothing
            grid_wh = parse(Int, m[1])
        elseif (m = match(r"n. timesteps  = ([0-9]+)", line)) !== nothing
            n_timesteps = parse(Int, m[1])
        elseif (m = match(r"Makespan: ([0-9]+\.[0-9]+) s", line)) !== nothing
            runtime = parse(Float64, m[1])
        end
    end

    metrics = build_metrics(
        (variant == :monotile) ? "FDTD, Monotile" : "FDTD, Tiling",
        runtime,
        n_timesteps * N_SUBGENERATIONS,
        variant,
        f,
        loop_latency,
        grid_wh,
        grid_wh,
        tile_height,
        tile_width,
        n_cus,
        OPERATIONS_PER_CELL,
        CELL_SIZE
    )
    
    open("metrics.$variant.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end
