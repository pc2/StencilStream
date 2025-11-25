#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using Glob

const MPI_ROOT = ENV["I_MPI_ROOT"]
const MPIRUN = "$MPI_ROOT/bin/mpirun"
const CELL_SIZE = 4 # bytes, including material coefficients
const MEMORY_SIZE = Dict(:monotile => 32*2^(30), :tiling => 32*2^(30), :cuda => 40*2^(30))

struct JacobiConfig
    variant::Symbol
    temporal_parallelism::Union{Int, Nothing}
    spatial_parallelism::Union{Int, Nothing}
    tile_height::Union{Int, Nothing}
    tile_width::Union{Int, Nothing}
    f::Union{Float64, Nothing}
    n_coefficients::Int
    n_operations_per_cell::Int
end

function load_config(exe)
    config_text = open(io -> join(readlines(io)), `$MPIRUN -n 1 $exe show-config`, "r")
    config = JSON.parse(match(r"(\{[^\}\{]+\})$", config_text)[1])
    variant = Symbol(config["variant"])
    JacobiConfig(
        variant,
        variant == :cuda ? nothing : config["temporal_parallelism"],
        variant == :cuda ? nothing : config["spatial_parallelism"],
        variant == :cuda ? nothing : config["tile_height"],
        variant == :cuda ? nothing : config["tile_width"],
        variant == :cuda ? nothing : load_report_details(exe * ".prj/reports"),
        config["n_coefficients"],
        config["n_operations"]
    )
end

function run_benchmark(exe, n_ranks, config::JacobiConfig, grid_wh, n_timesteps; n_samples=3, warmup=true)
    arguments = fill(1/config.n_coefficients, config.n_coefficients)
    command = `$exe $(grid_wh) $(grid_wh) $(n_timesteps) /dev/null $(arguments)`

    if config.variant == :monotile
        command = `$MPIRUN -n $n_ranks $command`
    end

    if warmup
        if config.variant == :monotile
            # Warmup to exclude programming from the benchmark
            warmup_cluster(command, n_ranks, config.variant)
        else
            run(command)
        end
    end

    # Defining names here so that later definitions won't be dropped.
    runtimes = Vector()
    for _ in 1:n_samples
        runtime = open(command, "r") do process_in
            runtime = nothing

            while !eof(process_in)
                line = readline(process_in)
                println(line)
                if (m = match(r"Walltime: ([0-9]+\.[0-9]+) s", line)) !== nothing
                    runtime = parse(Float64, m[1])
                end
            end

            runtime
        end
        push!(runtimes, runtime)
    end

    BenchmarkInformation(
        n_timesteps,
        grid_wh,
        grid_wh,
        n_ranks,
        1, # no. of subiterations
        CELL_SIZE,
        config.n_operations_per_cell,
        config.variant,
        config.temporal_parallelism,
        config.spatial_parallelism,
        config.tile_height,
        config.tile_width,
        config.f,
        minimum(runtimes)
    )
end

function max_perf_benchmark(exe, n_ranks, config::JacobiConfig)    
    if config.variant != :cuda
        grid_wh = min(config.tile_height, config.tile_width)
    else
        grid_wh = 8192
    end

    target_runtime = 15.0
    if config.variant == :cuda
        mem_throughput = 1.935e12 # Max bandwidth of an A100
        fp_throughput = 19.5e12 # Max FP32 throughput of an A100
        cell_size = 4
        target_total_updates = target_runtime * min(mem_throughput / cell_size, fp_throughput / config.n_operations_per_cell)
    else
        total_parallelism = config.temporal_parallelism * config.spatial_parallelism * n_ranks
        target_total_updates = target_runtime * total_parallelism * f
    end
    n_timesteps = target_total_updates / grid_wh^2
    if config.variant != :cuda
        n_timesteps = Int(config.temporal_parallelism * ceil(n_timesteps / config.temporal_parallelism)) # round up for max utilization
    else
        n_timesteps = Int(round(n_timesteps))
    end
    n_samples = 3

    info = run_benchmark(exe, n_ranks, config, grid_wh, n_timesteps; n_samples)

    exe_name = basename(exe)
    if config.variant == :cuda
        metrics = Dict(
            "target" => "$exe_name, CUDA",
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        target_name = config.variant == :monotile ? "Monotile" : "Tiling"
        metrics = Dict(
            "target" => "$exe_name, $target_name",
            "parallelity" => parallelity(info),
            "f" => info.f,
            "occupancy" => occupancy(info),
            "measured" => measured_throughput(info),
            "accuracy" => model_accurracy(info),
            "FLOPS" => measured_flops(info)
        )
    end

    open("metrics.$exe_name.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end

function deep_grid_scaling_benchmark(exec, n_ranks, config::JacobiConfig)
    if config.variant == :cuda || config.variant == :tiling
        # Maximal grid size that fits in global memory and is indexable with the 32-bit signed integers
        max_n_cells = min(MEMORY_SIZE[config.variant] / 3 / CELL_SIZE, 2^31)
        max_grid_wh = √(max_n_cells)
        # Round down to the next-lowest power of √2.
        # We scale in power-of-√2 steps, so that each grid is about half the size of the next-biggest
        grid_wh = (√2)^floor(log(√2, max_grid_wh))
    else
        grid_wh = config.tile_width
    end

    df_path = "scaling.$(config.variant).csv"
    df = DataFrame(grid_wh=Int64[], n_iters=Int64[], runtime=Float64[], measured_throughput=Float64[])

    first_iteration = true
    while grid_wh^2 >= 32
        true_grid_wh = Int(ceil(grid_wh))
        target_runtime = 30.0
        if config.variant == :monotile || config.variant == :tiling
            proto_info = BenchmarkInformation(
                n_ranks * config.temporal_parallelism, # Dummy number of iterations for one pass
                true_grid_wh,
                true_grid_wh,
                n_ranks,

                1, # No. of subiterations
                CELL_SIZE,
                config.n_operations_per_cell,

                config.variant,
                config.temporal_parallelism,
                config.spatial_parallelism,
                config.tile_height,
                config.tile_width,

                load_report_details(exec * ".prj/reports"), # Clock frequency

                42.0 # Dummy runtime
            )
            n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
            n_iters = n_ranks * config.temporal_parallelism * n_passes
        else
            mem_throughput = 1555.0 / 4 * 2^30
            cell_rate = mem_throughput / 2CELL_SIZE
            iteration_rate = cell_rate / true_grid_wh^2
            n_iters = Int(ceil(iteration_rate * target_runtime))
        end

        info = run_benchmark(exec, n_ranks, config, true_grid_wh, n_iters; n_samples=3, warmup=first_iteration)
        push!(df, [true_grid_wh, n_iters, info.runtime, measured_throughput(info)])
        CSV.write(df_path, df)

        grid_wh /= √2
        first_iteration = false
    end
end

if size(ARGS) != (3,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable or directory> <n_ranks>")
    println(stderr, "Possible benchmarks: max_perf deep_grid_scaling")
    exit(1)
end

exe = ARGS[2]
n_ranks = parse(Int, ARGS[3])
config = load_config(exe)

if ARGS[1] == "max_perf"
    max_perf_benchmark(exe, n_ranks, config)
elseif ARGS[1] == "deep_grid_scaling"
    deep_grid_scaling_benchmark(exe, n_ranks, config)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
