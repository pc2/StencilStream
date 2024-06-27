#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")
using DelimitedFiles
using Statistics

const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const N_MONOTILE_CUS = 280
const N_TILING_CUS = 224
const TILE_HEIGHT = 1024
const MONO_TILE_WIDTH = 1024
const TILING_TILE_WIDTH = 2^16

function create_experiment(n_rows, n_columns, temp_file, power_file)
    begin
        temp = fill(30.0f0, n_rows * n_columns)
        write(temp_file, temp)
    end

    begin
        power = zeros(Float32, n_rows, n_columns)
        power[(n_rows÷4):(3n_rows÷4), (n_columns÷4):(3n_columns÷4)] .= 0.5
        power = reshape(power', n_rows * n_columns)
        write(power_file, power)
    end
end

function max_perf_benchmark(exec, variant, f, loop_latency)
    if variant == :monotile
        n_cus = N_MONOTILE_CUS
        grid_height = 720
        grid_width = 1024
        n_iters = n_cus
        n_samples = 100
    elseif variant == :tiling
        n_cus = N_TILING_CUS
        grid_height = 16*1024
        grid_width = 16*1024
        n_iters = ceil(1000/n_cus) * n_cus
        n_samples = 3
    end

    experiment_dir = mktempdir("/dev/shm/")
    temp_path = experiment_dir * "/temp.bin"
    power_path = experiment_dir * "/power.bin"
    out_path = "/dev/null"
    println("temp: $temp_path, power: $power_path")

    println("Creating experiment...")
    create_experiment(grid_height, grid_width, temp_path, power_path)
    println("Experiment created and written!")

    command = `$exec $grid_height $grid_width $n_iters $temp_path $power_path $out_path`

    runtimes = Vector()
    for i_sample in 1:n_samples
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
        push!(runtimes, runtime)
    end
    runtime = mean(runtimes)
    tile_width = (variant == :monotile) ? MONO_TILE_WIDTH : TILING_TILE_WIDTH

    info = BenchmarkInformation(
        n_iters,
        grid_height,
        grid_width,
        1,
        CELL_SIZE,
        OPERATIONS_PER_CELL,
        variant,
        n_cus,
        TILE_HEIGHT,
        tile_width,
        f,
        loop_latency,
        runtime
    )

    metrics = Dict(
        "target" => (variant == :monotile) ? "Hotspot, Monotile" : "Hotspot, Tiling",
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

function scaling_benchmark(exec, variant, f, loop_latency)
    out_path = "$(variant)_perf.csv"

    temp_path, temp_io = mktemp()
    power_path, power_io = mktemp()
    close(temp_io)
    close(power_io)

    # Run the simulation once to eliminate the FPGA programming from the measured runtime
    run(`$exec 1024 1024 1024 ./data/temp_1024 ./data/power_1024 /dev/null`)

    df = DataFrame(grid_wh=Int64[], n_timesteps=Int64[], kernel_runtime=Float64[], walltime=Float64[], model_runtime=Float64[])

    for grid_wh in 128:128:1024
        create_experiment(grid_wh, grid_wh, temp_path, power_path)

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

            info = BenchmarkInformation(
                n_timesteps,
                grid_wh,
                grid_wh,
                1,
                CELL_SIZE,
                OPERATIONS_PER_CELL,
                variant,
                (variant == :monotile) ? N_MONOTILE_CUS : N_TILING_CUS,
                TILE_HEIGHT,
                (variant == :monotile) ? MONO_TILE_WIDTH : TILING_TILE_WIDTH,
                f,
                loop_latency,
                kernel_runtime
            )

            push!(df, (grid_wh, n_timesteps, kernel_runtime, walltime, model_runtime(info)))
            CSV.write(out_path, df)
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

variant = Symbol(ARGS[3])

if ARGS[1] == "max_perf"
    max_perf_benchmark(exec, variant, f, loop_latency)
elseif ARGS[1] == "scaling"
    scaling_benchmark(exec, variant, f, loop_latency)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end