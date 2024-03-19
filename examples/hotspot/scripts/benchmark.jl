#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")
using DelimitedFiles

const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const N_MONOTILE_CUS = 280
const N_TILING_CUS = 224
const TILE_HEIGHT = 1024
const MONO_TILE_WIDTH = 1024
const TILING_TILE_WIDTH = 2^16

function write_outputs(n, temp_file, power_file; width=0.1)
    temp = fill(30.0, n^2)
    power = Matrix{Float64}(undef, n, n)
    middle = n ÷ 2
    radius = round((width * n) / 2)
    for I in CartesianIndices(power)
        if all(middle - radius .<= Tuple(I) .<= middle + radius)
            power[I] = 0.5
        else
            power[I] = 0.0
        end
    end
    power = reshape(power, n^2)
    writedlm(temp_file, temp)
    writedlm(power_file, power)
end

function max_perf_benchmark(exec, variant, n_cus, f, loop_latency)
    if variant == :monotile
        grid_size = TILE_SIZE
    elseif variant == :tiling
        grid_size = 3TILE_SIZE
    end

    temp_path, temp_io = mktemp()
    power_path, power_io = mktemp()
    close(temp_io)
    close(power_io)
    write_outputs(grid_size, temp_path, power_path)

    n_gens = n_cus * 1000
    command = `$exec $grid_size $grid_size $n_gens $temp_path $power_path /dev/null`

    # Run the simulation once to eliminate the FPGA programming from the measured runtime
    run(command)

    runtime = open(command, "r") do process_in
        while true
            line = readline(process_in)
            println(line)
    
            line_match = match(r"Kernel Runtime: ([0-9]+\.[0-9]+) s", line)
            if line_match !== nothing
                return parse(Float64, line_match[1])
            end
        end
    end
    tile_width = (variant == :monotile) ? MONO_TILE_WIDTH : TILING_TILE_WIDTH
    raw_metrics = build_metrics(runtime, n_gens, variant, f, loop_latency, grid_size, grid_size, TILE_HEIGHT, tile_width, n_cus, OPERATIONS_PER_CELL, CELL_SIZE)

    metrics = Dict(
        "target" => (variant == :monotile) ? "Hotspot, Monotile" : "Hotspot, Tiling",
        "n_cus" => n_cus,
        "f" => f,
        "occupancy" => raw_metrics[:occupancy],
        "measured" => raw_metrics[:measured_rate],
        "accuracy" => raw_metrics[:model_accurracy],
        "FLOPS" => raw_metrics[:flops],
        "mem_throughput" => raw_metrics[:mem_throughput]
    )
    
    open("metrics.$variant.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end

    rm(temp_path)
    rm(power_path)
end

function scaling_benchmark(exec, variant)
    out_path = "$(variant)_perf.csv"

    temp_path, temp_io = mktemp()
    power_path, power_io = mktemp()
    close(temp_io)
    close(power_io)

    # Run the simulation once to eliminate the FPGA programming from the measured runtime
    run(`$exec 1024 1024 1024 ./data/temp_1024 ./data/power_1024 /dev/null`)

    df = DataFrame(grid_wh=Int64[], n_timesteps=Int64[], kernel_runtime=Float64[], walltime=Float64[], model_runtime=Float64[])

    for iteration in 1:1
        for grid_wh in 128:128:1024
            write_outputs(grid_wh, temp_path, power_path)

            for n_timesteps in 50_000:50_000:1_000_000
                command = `$exec $grid_wh $grid_wh $n_timesteps $temp_path $power_path /dev/null`
                kernel_runtime, walltime = open(command, "r") do process_in
                    kernel_runtime = nothing
                    walltime = nothing
                    while any(v -> v === nothing, [kernel_runtime, walltime])
                        line = readline(process_in)
                        println(line)

                        if (m = match(r"Kernel Runtime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                            kernel_runtime = parse(Float64, m[1])
                        elseif (m = match(r"Walltime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                            walltime = parse(Float64, m[1])
                        end
                    end
                    kernel_runtime, walltime
                end

                if variant == :monotile
                    model_runtime = model_monotile_runtime(f, loop_latency, grid_wh, grid_wh, n_timesteps, N_MONOTILE_CUS)
                else
                    model_runtime = model_tiling_runtime(f, loop_latency, grid_wh, grid_wh, n_timesteps, TILE_HEIGHT, TILING_TILE_WIDTH, N_TILING_CUS)
                end

                push!(df, (grid_wh, n_timesteps, kernel_runtime, walltime, model_runtime))
                CSV.write(out_path, df)
            end
        end
    end

    render_model_error(df, "$(variant)_relative_model_error.mp4")

    rm(temp_path)
    rm(power_path)
end

if size(ARGS) != (3,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant>")
    println(stderr, "Possible benchmarks: max_perf, scaling")
    println(stderr, "Possible variants: monotile, tiling")
    exit(1)
end

exec = ARGS[2]
report_path = exec * ".prj/reports"
f, loop_latency = load_report_details(report_path)

variant = ARGS[3]
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

if ARGS[1] == "max_perf"
    max_perf_benchmark(exec, variant, n_cus, f, loop_latency)
elseif ARGS[1] == "scaling"
    scaling_benchmark(exec, variant)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end