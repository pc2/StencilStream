#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 3
const OPERATIONS_PER_CELL = (5 + 5 + 3 + 6 + 6 + 6) + (10 + 3 + 2 + 14 + 3 + 2) + 2
const CELL_SIZE = 88 # bytes
const TEMPORAL_PARALLELISM = Dict(:mono => 6, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:mono => 1, :cuda => 1)
const TILE_NX = Dict(:mono => 2^16, :cuda => nothing)
const TILE_NY = Dict(:mono => 512, :cuda => nothing)

function analyze_log(logfile)
    iteration_re = r"it = ([0-9]+) \(iter = ([0-9]+), time = ([^)]+)\)"
    computation_time_re = r"Of which transient computation time: (.+) s$"

    computation_time = nothing
    n_iterations = 0

    while !eof(logfile)
        line = readline(logfile)
        if length(line) == 0
            continue
        end
        println(line)

        if (line_match = match(iteration_re, line)) !== nothing
            n_iterations += parse(Int, line_match[2])

        elseif (line_match = match(computation_time_re, line)) !== nothing
            computation_time = parse(Float64, line_match[1])
            
        end
    end

    return computation_time, n_iterations
end

function max_perf_benchmark(exe, variant)
    if variant == :cuda
        experiment_path = "experiments/cuda-benchmark.json"
    else
        experiment_path = "experiments/max-res-default.json"
    end
    experiment_data = JSON.parsefile(experiment_path)
    lx = experiment_data["lx"]
    ly = experiment_data["ly"]
    res = experiment_data["res"]

    out_path = Base.Filesystem.mkpath("./out/")
    command = `$exe $experiment_path $out_path`

    open(command, "r") do process_in
        computation_time, n_iterations = analyze_log(process_in)

        info = BenchmarkInformation(
            n_iterations,
            lx * res,
            ly * res,
            nothing, # No multi-FPGA usage
            N_SUBITERATIONS,
            CELL_SIZE,
            OPERATIONS_PER_CELL,
            variant,
            TEMPORAL_PARALLELISM[variant],
            SPATIAL_PARALLELISM[variant],
            TILE_NX[variant],
            TILE_NY[variant],
            (variant == :mono) ? load_report_details(exe * ".prj/reports") : nothing,
            computation_time
        )

        if variant == :cuda
            metrics = Dict(
                "target" => "Convection, CUDA",
                "measured" => measured_throughput(info),
                "FLOPS" => measured_flops(info)
            )
        else
            metrics = Dict(
                "target" => "Convection, Monotile",
                "parallelity" => parallelity(info),
                "f" => info.f,
                "occupancy" => occupancy(info),
                "measured" => measured_throughput(info),
                "accuracy" => model_accurracy(info),
                "FLOPS" => measured_flops(info)
            )
        end

        open("metrics.$(variant).json", "w") do metrics_file
            JSON.print(metrics_file, metrics)
        end
    end
end

function print_usage()
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_fpgas>")
    println(stderr, "Possible benchmarks: max_perf")
    println(stderr, "Possible variants: mono, cuda")
    exit(1)
end

if size(ARGS) != (4,)
    print_usage()
end

mode = ARGS[1]
exe = ARGS[2]
variant = Symbol(ARGS[3])
n_ranks = parse(Int, ARGS[4])
if variant ∉ [:mono, :cuda]
    println(stderr, "Unsupported variant $variant")
    println(stderr)
    print_usage()
end

if mode == "max_perf"
    max_perf_benchmark(exe, variant)
end
