#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 2
const N_TILING_CUS = 190
const N_MONOTILE_CUS = 200
const OPERATIONS_PER_CELL = 8 + (6 + 4 + 2 + 2 + 2) # Including all paths, excluding source wave computation
const CELL_SIZE = 4 * (4 + 4) # bytes, including material coefficients
const TILE_HEIGHT = 512
const MONO_TILE_WIDTH = 512
const TILING_TILE_WIDTH = 2^16

function max_perf_benchmark(exe, variant, f, loop_latency)
    if variant == :monotile
        experiment_path = "./experiments/full_tile.json"
        tile_height = TILE_HEIGHT
        tile_width = MONO_TILE_WIDTH
        n_cus = N_MONOTILE_CUS
    elseif variant == :tiling
        experiment_path = "./experiments/max_grid.json"
        tile_height = TILE_HEIGHT
        tile_width = TILING_TILE_WIDTH
        n_cus = N_TILING_CUS
    end
    command = `$exe -c $experiment_path`

    kernel_runtime, walltime, grid_wh, n_timesteps = open(command, "r") do process_in
        kernel_runtime = nothing
        walltime = nothing
        grid_wh = nothing
        n_timesteps = nothing

        # While any of the metrics is nothing...
        while any(v -> v === nothing, [kernel_runtime, walltime, grid_wh, n_timesteps])
            line = readline(process_in)
            println(line)
            if (m = match(r"grid w/h\s*= ([0-9]+) cells", line)) !== nothing
                grid_wh = parse(Int, m[1])
            elseif (m = match(r"n. timesteps\s*= ([0-9]+)", line)) !== nothing
                n_timesteps = parse(Int, m[1])
            elseif (m = match(r"Kernel Runtime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                kernel_runtime = parse(Float64, m[1])
            elseif (m = match(r"Walltime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                walltime = parse(Float64, m[1])
            end
        end

        kernel_runtime, walltime, grid_wh, n_timesteps
    end
    info = BenchmarkInformation(
        n_timesteps,
        grid_wh,
        grid_wh,
        N_SUBITERATIONS,
        CELL_SIZE,
        OPERATIONS_PER_CELL,
        variant,
        n_cus,
        tile_height,
        tile_width,
        f,
        loop_latency,
        kernel_runtime
    )

    metrics = Dict(
        "target" => (variant == :monotile) ? "FDTD, Monotile" : "FDTD, Tiling",
        "n_cus" => n_cus,
        "f" => f,
        "occupancy" => occupancy(info),
        "measured" => measured_throughput(info),
        "accuracy" => model_accurracy(info),
        "FLOPS" => measured_flops(info),
        "mem_throughput" => measured_mem_throughput(info)
    )

    open("metrics.$variant.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end

function scaling_benchmark(exe, variant, f, loop_latency)
    mkpath("out/")
    out_path = "$(variant)_perf.csv"

    # Run the simulation once to eliminate the FPGA programming from the measured runtime
    run(`$exe -c ./experiments/default.json -o out/`)

    experiment = JSON.parsefile("experiments/full_tile.json")
    max_width = experiment["cavity_rings"][1]["width"]
    df = DataFrame(t_max=Float64[], grid_wh=Int64[], n_timesteps=Int64[], kernel_runtime=Float64[], walltime=Float64[], model_runtime=Float64[])
    experiment_path, experiment_io = mktemp()
    close(experiment_io)

    if variant == :monotile
        tile_height = TILE_HEIGHT
        tile_width = MONO_TILE_WIDTH
        n_cus = N_MONOTILE_CUS
    elseif variant == :tiling
        tile_height = TILE_HEIGHT
        tile_width = TILING_TILE_WIDTH
        n_cus = N_TILING_CUS
    end

    for iteration in 1:3
        for rel_width in 0.1:0.1:1.0
            for t_max in 1.0:1.0:30.0
                experiment["time"]["t_max"] = t_max
                experiment["cavity_rings"][1]["width"] = rel_width * max_width
                open(io -> JSON.print(io, experiment), experiment_path, "w")

                kernel_runtime, walltime, grid_wh, n_timesteps = open(`$exe -c $experiment_path -o out/`) do process_in
                    kernel_runtime = nothing
                    walltime = nothing
                    grid_wh = nothing
                    n_timesteps = nothing

                    # While any of the metrics is nothing...
                    while any(v -> v === nothing, [kernel_runtime, walltime, grid_wh, n_timesteps])
                        line = readline(process_in)
                        println(line)
                        if (m = match(r"grid w/h      = ([0-9]+) cells", line)) !== nothing
                            grid_wh = parse(Int, m[1])
                        elseif (m = match(r"n. timesteps  = ([0-9]+)", line)) !== nothing
                            n_timesteps = parse(Int, m[1])
                        elseif (m = match(r"Kernel Runtime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                            kernel_runtime = parse(Float64, m[1])
                        elseif (m = match(r"Walltime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                            walltime = parse(Float64, m[1])
                        end
                    end

                    kernel_runtime, walltime, grid_wh, n_timesteps
                end

                info = BenchmarkInformation(
                    n_timesteps,
                    grid_wh,
                    grid_wh,
                    N_SUBITERATIONS,
                    CELL_SIZE,
                    OPERATIONS_PER_CELL,
                    variant,
                    n_cus,
                    tile_height,
                    tile_width,
                    f,
                    loop_latency,
                    kernel_runtime
                )

                push!(df, (t_max, grid_wh, n_timesteps, kernel_runtime, walltime, model_runtime(info)))
                CSV.write(out_path, df)
            end
        end
    end

    render_model_error(df, "$(variant)_relative_model_error.mp4")

    rm(experiment_path)
end

if size(ARGS) != (3,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant>")
    println(stderr, "Possible benchmarks: max_perf, scaling")
    println(stderr, "Possible variants: monotile, tiling")
    exit(1)
end

exe = ARGS[2]
report_path = exe * ".prj/reports"
f, loop_latency = load_report_details(report_path)

variant = Symbol(ARGS[3])

if ARGS[1] == "max_perf"
    max_perf_benchmark(exe, variant, f, loop_latency)
elseif ARGS[1] == "scaling"
    scaling_benchmark(exe, variant, f, loop_latency)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end