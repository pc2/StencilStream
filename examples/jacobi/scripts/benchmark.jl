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
    temporal_parallelism = config["temporal_parallelism"]
    spatial_parallelism = config["spatial_parallelism"]
    tile_height = config["tile_height"]
    tile_width = config["tile_width"]
    n_coefficients = config["n_coefficients"]
    n_operations_per_cell = config["n_operations"]
    f = load_report_details(exe * ".prj/reports")

    grid_wh = min(tile_height, tile_width)

    # target runtime = 15s.
    target_total_updates = 15 * temporal_parallelism * spatial_parallelism * n_ranks * f
    n_timesteps = target_total_updates / grid_wh^2
    n_timesteps = Int(temporal_parallelism * ceil(n_timesteps / temporal_parallelism)) # round up for max utilization
    n_samples = 3

    arguments = fill(1/n_coefficients, n_coefficients)

    mpi_root = ENV["I_MPI_ROOT"]
    command = `$mpirun -n $n_ranks $exe $(grid_wh) $(grid_wh) $(n_timesteps) /dev/null $(arguments)`

    # Warmup to exclude programming from the benchmark
    warmup_cluster(command, n_ranks, variant; links_preconfigured=true)

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

    metrics = Dict(
        "target" => exe_name,
        "parallelity" => parallelity(info),
        "f" => info.f,
        "occupancy" => occupancy(info),
        "measured" => measured_throughput(info),
        "accuracy" => model_accurracy(info),
        "FLOPS" => measured_flops(info)
    )

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
    # TODO: Extend to tiling
    setup_io_pipes(n_ranks, :monotile)
    if isdir(exe_or_dir)
        for exe in glob("$exe_or_dir/Jacobi*_mono")
            max_perf_benchmark(exe, n_ranks)
        end
    else
        max_perf_benchmark(exe_or_dir, n_ranks)
    end
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
