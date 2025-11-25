#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using JSON

const N_SUBITERATIONS = 2
const OPERATIONS_PER_CELL = 8 + (6 + 4 + 2 + 2 + 2) # Including all paths, excluding source wave computation
const CELL_SIZE = 4 * (4 + 4) # bytes, including material coefficients
const TEMPORAL_PARALLELISM = Dict(:monotile => 100, :tiling => 52, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:monotile => 1, :tiling => 2, :cuda => 1)
const TILE_HEIGHT = Dict(:monotile => 512, :tiling => 2^16, :cuda => nothing)
const TILE_WIDTH = Dict(:monotile => 512, :tiling => 768, :cuda => nothing)
const MEMORY_SIZE = Dict(:monotile => 32*2^(30), :tiling => 32*2^(30), :cuda => 40*2^(30))

function run_benchmark(exe, variant, n_ranks, experiment_path; n_samples=3)
    out_dir = Base.Filesystem.mkpath("./out/")
    command = `$exe -c $experiment_path -o $out_dir`

    if variant == :monotile
        mpi_root = ENV["I_MPI_ROOT"]
        command = `$mpi_root/bin/mpirun -n $n_ranks $exe -c $experiment_path -o $out_dir`
        warmup_cluster(command, n_ranks, variant)
    end

    runtime_re = Regex("Walltime: ([0-9]+\\.[0-9]+) s")

    # Defining names here so that later definitions won't be dropped.
    grid_wh = n_timesteps = nothing
    runtimes = Vector()

    for _ in 1:n_samples
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

    BenchmarkInformation(
        n_timesteps,
        grid_wh,
        grid_wh,
        n_ranks,
        N_SUBITERATIONS,
        CELL_SIZE,
        OPERATIONS_PER_CELL,
        variant,
        TEMPORAL_PARALLELISM[variant],
        SPATIAL_PARALLELISM[variant],
        TILE_HEIGHT[variant],
        TILE_WIDTH[variant],
        (variant == :cuda) ? nothing : load_report_details(exe * ".prj/reports"),
        minimum(runtimes)
    )
end

function max_perf_benchmark(exe, variant, n_ranks)
    if variant == :monotile 
        experiment_path = "./experiments/mono_benchmark.json"
        n_samples = 10
    elseif variant == :tiling
        experiment_path = "./experiments/tiling_benchmark.json"
        n_samples = 3
    elseif variant == :cuda
        experiment_path = "./experiments/cuda_benchmark.json"
        n_samples = 10
    end
    
    info = run_benchmark(exe, variant, n_ranks, experiment_path; n_samples)

    if variant == :cuda
        metrics = Dict(
            "target" => "FDTD, CUDA",
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        metrics = Dict(
            "target" => (variant == :monotile) ? "FDTD, Monotile" : "FDTD, Tiling",
            "parallelity" => parallelity(info),
            "f" => info.f,
            "occupancy" => occupancy(info),
            "measured" => measured_throughput(info),
            "accuracy" => model_accurracy(info),
            "FLOPS" => measured_flops(info)
        )
    end

    metrics
end

const c0 = 299792458.0
experiment_grid_wh(experiment) = Int(round(2.0 * experiment["cavity_rings"][1]["radius"] / experiment["dx"])) + 2
experiment_dt(experiment) = (experiment["dx"] / (c0 * √2)) * 0.99
experiment_n_iters(experiment) = ceil((experiment["time"]["t_max"] * experiment["tau"]) / experiment_dt(experiment))

dx_for_grid_wh(grid_wh, experiment) = 2.0 * experiment["cavity_rings"][1]["radius"] / (grid_wh - 2)
t_max_for_n_iters(n_iters, experiment) = (n_iters * experiment_dt(experiment)) / (experiment["tau"])

function deep_grid_scaling_benchmark(exec, variant, n_ranks)
    experiment = open(JSON.parse, "./experiments/mono_benchmark.json")
    if variant == :cuda || variant == :tiling
        max_grid_wh = √(MEMORY_SIZE[variant] / 3 / CELL_SIZE) # Maximal grid size that fits in global memory
        # Round down to the next-lowest power of √2.
        # We scale in power-of-√2 steps, so that each grid is about half the size of the next-biggest
        grid_wh = (√2)^floor(log(√2, max_grid_wh))
    else
        grid_wh = TILE_WIDTH[:monotile]
    end

    experiment_path, _ = mktemp()

    df = DataFrame(grid_wh=Int64[], n_iters=Int64[], runtime=Float64[], measured_throughput=Float64[])

    while grid_wh >= 8
        experiment["dx"] = dx_for_grid_wh(round(grid_wh), experiment)

        target_runtime = 30.0
        if variant == :monotile || variant == :tiling
            proto_info = BenchmarkInformation(
                n_ranks * TEMPORAL_PARALLELISM[variant], # Dummy number of iterations for one pass
                round(grid_wh),
                round(grid_wh),
                n_ranks,

                1, # No. of subiterations
                CELL_SIZE,
                OPERATIONS_PER_CELL,

                variant,
                TEMPORAL_PARALLELISM[variant],
                SPATIAL_PARALLELISM[variant],
                TILE_HEIGHT[variant],
                TILE_WIDTH[variant],

                load_report_details(exec * ".prj/reports"), # Clock frequency

                42.0 # Dummy runtime
            )
            n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
            n_iters = n_ranks * TEMPORAL_PARALLELISM[variant] * n_passes
        else
            mem_throughput = 1555.0 * 2^30
            cell_rate = mem_throughput / 2CELL_SIZE
            iteration_rate = cell_rate / round(grid_wh)^2 / N_SUBITERATIONS
            n_iters = Int(ceil(iteration_rate * target_runtime))
        end
        experiment["time"]["t_max"] = t_max_for_n_iters(n_iters, experiment)

        open(f -> JSON.print(f, experiment), experiment_path, "w")

        info = run_benchmark(exec, variant, n_ranks, experiment_path; n_samples=3)
        push!(df, [info.n_grid_rows, info.n_iters, info.runtime, measured_throughput(info)])
        CSV.write("scaling.$variant.csv", df)

        grid_wh /= √2
    end
end

if size(ARGS) != (4,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_fpgas>")
    println(stderr, "Possible benchmarks: max_perf")
    println(stderr, "Possible variants: monotile, tiling, cuda")
    exit(1)
end

exe = ARGS[2]
variant = Symbol(ARGS[3])
n_ranks = parse(Int, ARGS[4])

if ARGS[1] == "max_perf"
    metrics = max_perf_benchmark(exe, variant, n_ranks)
    open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
elseif ARGS[1] == "strong_scaling"
    metrics = Dict()
    for i in n_ranks:-1:1
        metrics[i] = max_perf_benchmark(exe, variant, i)
        open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
    end
elseif ARGS[1] == "deep_grid_scaling"
    deep_grid_scaling_benchmark(exe, variant, n_ranks)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
