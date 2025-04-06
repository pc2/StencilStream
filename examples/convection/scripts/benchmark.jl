#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 3
const OPERATIONS_PER_CELL = (5 + 5 + 3 + 6 + 6 + 6) + (10 + 3 + 2 + 14 + 3 + 2) + 2
const CELL_SIZE = 88 # bytes
const TEMPORAL_PARALLELISM = Dict(:monotile => 8, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:monotile => 1, :cuda => 1)
const TILE_NX = Dict(:monotile => 2^16, :cuda => nothing)
const TILE_NY = Dict(:monotile => 768, :cuda => nothing)

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
    experiment_path = "experiments/max-res-default.json"
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
            N_SUBITERATIONS,
            CELL_SIZE,
            OPERATIONS_PER_CELL,
            variant,
            TEMPORAL_PARALLELISM[variant],
            SPATIAL_PARALLELISM[variant],
            TILE_NX[variant],
            TILE_NY[variant],
            (variant == :monotile) ? load_report_details(exe * ".prj/reports") : nothing,
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
                "n_cus" => n_replications(info),
                "f" => info.f,
                "occupancy" => occupancy(info),
                "measured" => measured_throughput(info),
                "accuracy" => model_accurracy(info),
                "FLOPS" => measured_flops(info),
                "mem_throughput" => measured_mem_throughput(info)
            )
        end

        open("metrics.$(variant).json", "w") do metrics_file
            JSON.print(metrics_file, metrics)
        end
    end
end

mode = ARGS[1]
exe = ARGS[2]
variant = Symbol(ARGS[3])
if mode == "max_perf"
    max_perf_benchmark(exe, variant)
end
