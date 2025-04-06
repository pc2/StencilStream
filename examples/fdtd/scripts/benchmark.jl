#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 2
const OPERATIONS_PER_CELL = 8 + (6 + 4 + 2 + 2 + 2) # Including all paths, excluding source wave computation
const CELL_SIZE = 4 * (4 + 4) # bytes, including material coefficients
const TEMPORAL_PARALLELISM = Dict(:monotile => 72, :tiling => 64, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:monotile => 2, :tiling => 2, :cuda => 1)
const TILE_HEIGHT = Dict(:monotile => 1024, :tiling => 2^16, :cuda => nothing)
const TILE_WIDTH = Dict(:monotile => 1024, :tiling => 768, :cuda => nothing)

function max_perf_benchmark(exe, variant)
    if variant == :monotile || variant == :cuda
        experiment_path = "./experiments/mono_benchmark.json"
        n_samples = 10
    elseif variant == :tiling
        experiment_path = "./experiments/tiling_benchmark.json"
        n_samples = 3
    end
    out_dir = Base.Filesystem.mkpath("./out/")
    command = `$exe -c $experiment_path -o $out_dir`

    runtime_re = Regex("Walltime: ([0-9]+\\.[0-9]+) s")

    # Defining names here so that later definitions won't be dropped.
    grid_wh = n_timesteps = nothing
    runtimes = Vector()

    # Warmup to exclude programming from the benchmark
    run(command)

    for i_sample in 1:n_samples
        runtime, grid_wh, n_timesteps = open(command, "r") do process_in
            runtime = nothing
            grid_wh = nothing
            n_timesteps = nothing

            while !eof(process_in)
                line = readline(process_in)
                println(line)
                if (m = match(r"grid w/h\s*= ([0-9]+) cells", line)) !== nothing
                    grid_wh = parse(Int, m[1])
                elseif (m = match(r"n. timesteps\s*= ([0-9]+)", line)) !== nothing
                    n_timesteps = parse(Int, m[1])
                elseif (m = match(runtime_re, line)) !== nothing
                    runtime = parse(Float64, m[1])
                end
            end

            runtime, grid_wh, n_timesteps
        end
        push!(runtimes, runtime)
    end

    info = BenchmarkInformation(
        n_timesteps,
        grid_wh,
        grid_wh,
        N_SUBITERATIONS,
        CELL_SIZE,
        OPERATIONS_PER_CELL,
        variant,
        TEMPORAL_PARALLELISM[variant],
        SPATIAL_PARALLELISM[variant],
        TILE_HEIGHT[variant],
        TILE_WIDTH[variant],
        (variant == :cuda) ? nothing : load_report_details(exe * ".prj/reports"),
        mean(runtimes)
    )

    if variant == :cuda
        metrics = Dict(
            "target" => "FDTD, CUDA",
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        metrics = Dict(
            "target" => (variant == :monotile) ? "FDTD, Monotile" : "FDTD, Tiling",
            "n_cus" => n_replications(info),
            "f" => info.f,
            "occupancy" => occupancy(info),
            "measured" => measured_throughput(info),
            "accuracy" => model_accurracy(info),
            "FLOPS" => measured_flops(info),
            "mem_throughput" => measured_mem_throughput(info)
        )
    end

    open("metrics.$variant.json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end

if size(ARGS) != (3,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant>")
    println(stderr, "Possible benchmarks: max_perf")
    println(stderr, "Possible variants: monotile, tiling, cuda")
    exit(1)
end

exe = ARGS[2]
variant = Symbol(ARGS[3])

if ARGS[1] == "max_perf"
    max_perf_benchmark(exe, variant)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end