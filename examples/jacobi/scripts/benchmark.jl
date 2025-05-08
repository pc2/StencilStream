#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")

function max_perf_benchmark(exe)
    exe_name = basename(exe)
    
    config = open(io -> JSON.parse(join(readlines(io))), `$exe show-config`, "r")
    variant = Symbol(config["variant"])
    temporal_parallelism = config["temporal_parallelism"]
    spatial_parallelism = config["spatial_parallelism"]
    tile_height = config["tile_height"]
    tile_width = config["tile_width"]
    n_coefficients = config["n_coefficients"]
    n_operations_per_cell = config["n_operations"]

    grid_wh = min(tile_height, tile_width)

    # target runtime = 60s. Designs reach roughly 2 TFLOPS, so total operations are given as below:
    target_total_operations = 2e12 * 60
    n_timesteps = target_total_operations / n_operations_per_cell / grid_wh^2
    n_timesteps = Int(temporal_parallelism * ceil(n_timesteps / temporal_parallelism)) # round up for max utilization
    n_samples = 10

    arguments = fill(1/n_coefficients, n_coefficients)

    command = `$exe $(grid_wh) $(grid_wh) $(n_timesteps) /dev/null $(arguments)`

    # Warmup to exclude programming from the benchmark
    run(`$exe 1024 1024 1024 /dev/null $(arguments)`)

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
        1, # no. of subiterations
        4, # cell size
        n_operations_per_cell,
        variant,
        temporal_parallelism,
        spatial_parallelism,
        tile_height,
        tile_width,
        load_report_details(exe * ".prj/reports"),
        mean(runtimes)
    )

    metrics = Dict(
        "target" => exe_name,
        "n_cus" => n_replications(info),
        "f" => info.f,
        "occupancy" => occupancy(info),
        "measured" => measured_throughput(info),
        "accuracy" => model_accurracy(info),
        "FLOPS" => measured_flops(info),
        "mem_throughput" => measured_mem_throughput(info)
    )

    open("metrics.$exe_name.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end

if size(ARGS) != (2,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> ")
    println(stderr, "Possible benchmarks: max_perf")
    exit(1)
end

exe = ARGS[2]

if ARGS[1] == "max_perf"
    max_perf_benchmark(exe)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end