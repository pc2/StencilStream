#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using Glob

function max_perf_benchmark(exe, n_ranks)
    exe_name = basename(exe)

    mpi_root = ENV["I_MPI_ROOT"]
    mpirun = "$mpi_root/bin/mpirun"
    
    config_text = open(io -> join(readlines(io)), `$mpirun -n 1 $exe show-config`, "r")
    config = JSON.parse(match(r"(\{[^\}\{]+\})$", config_text)[1])
    variant = Symbol(config["variant"])
    if variant != :cuda
        temporal_parallelism = config["temporal_parallelism"]
        spatial_parallelism = config["spatial_parallelism"]
        tile_height = config["tile_height"]
        tile_width = config["tile_width"]
        grid_wh = min(tile_height, tile_width)
        f = load_report_details(exe * ".prj/reports")
    else
        temporal_parallelism = 1
        spatial_parallelism = 1
        tile_height = nothing
        tile_width = nothing
        grid_wh = 8192
        f = nothing
    end
    n_coefficients = config["n_coefficients"]
    n_operations_per_cell = config["n_operations"]

    # target runtime = 15s.
    target_runtime = 15.0
    if variant == :cuda
        mem_throughput = 1.935e12 # Max bandwidth of an A100
        fp_throughput = 19.5e12 # Max FP32 throughput of an A100
        cell_size = 4
        target_total_updates = target_runtime * min(mem_throughput / cell_size, fp_throughput / n_operations_per_cell)
    else
        total_parallelism = temporal_parallelism * spatial_parallelism * n_ranks
        target_total_updates = target_runtime * total_parallelism * f
    end
    n_timesteps = target_total_updates / grid_wh^2
    n_timesteps = Int(temporal_parallelism * ceil(n_timesteps / temporal_parallelism)) # round up for max utilization
    n_samples = 3

    arguments = fill(1/n_coefficients, n_coefficients)
    command = `$exe $(grid_wh) $(grid_wh) $(n_timesteps) /dev/null $(arguments)`

    # Set up the multi-FPGA cluster
    if variant == :multi_mono
        command = `$mpirun -n $n_ranks $command`

        # Warmup to exclude programming from the benchmark
        warmup_cluster(command, n_ranks, variant)
    end

    # Defining names here so that later definitions won't be dropped.
    runtimes = Vector()
    for i_sample in 1:n_samples
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

    info = BenchmarkInformation(
        n_timesteps,
        grid_wh,
        grid_wh,
        n_ranks,
        1, # no. of subiterations
        4, # cell size
        n_operations_per_cell,
        variant,
        temporal_parallelism,
        spatial_parallelism,
        tile_height,
        tile_width,
        f,
        mean(runtimes)
    )

    name_components = match(r"Jacobi([0-9])((Constant)|(General))", basename(exe_name))
    points = parse(Int, name_components[1])
    coef_type = name_components[2]
    if variant == :mono
        backend = "Single-FPGA Monotile"
    elseif variant == :multi_mono
        backend = "Multi-FPGA Monotile"
    elseif variant == :tiling
        backend = "Tiling"
    elseif variant == :cuda
        backend = "CUDA"
    end
    target_name = "$(points)-point $coef_type Jacobi, $backend"

    if variant == :cuda
        metrics = Dict(
            "target" => target_name,
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        metrics = Dict(
            "target" => target_name,
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

if size(ARGS) != (3,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable or directory> <n_ranks>")
    println(stderr, "Possible benchmarks: max_perf")
    exit(1)
end

exe_or_dir = ARGS[2]
n_ranks = parse(Int, ARGS[3])

if ARGS[1] == "max_perf"
    max_perf_benchmark(exe_or_dir, n_ranks)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
