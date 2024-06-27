#!/usr/bin/env -S julia --project=../..
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 3
const N_REPLICATIONS = 8
const N_CUS = N_SUBITERATIONS * N_REPLICATIONS
const OPERATIONS_PER_CELL = (5 + 5 + 3 + 6 + 6 + 6) + (10 + 3 + 2 + 14 + 3 + 2) + 2
const CELL_SIZE = 88 # bytes
const TILE_HEIGHT = 512
const TILE_WIDTH = 2^16

function analyze_log(logfile)
    iteration_re = r"it = ([0-9]+) \(iter = ([0-9]+), time = ([^)]+)\)"
    total_re = r"Total time = (.+)$"

    e2e_runtime = nothing
    pseudo_transient_runtimes = DataFrame(i_iteration=Int[], pseudo_steps=Int[], runtime=Float64[])

    while e2e_runtime === nothing
        line = readline(logfile)
        println(line)
        if length(line) == 0
            continue
        end

        if (line_match = match(iteration_re, line)) !== nothing
            i_iteration = parse(Int, line_match[1])
            pseudo_steps = parse(Int, line_match[2])
            runtime = parse(Float64, line_match[3])

            push!(pseudo_transient_runtimes, (i_iteration, pseudo_steps, runtime))
        elseif (line_match = match(total_re, line)) !== nothing
            e2e_runtime = parse(Float64, line_match[1])

        else
            println(stderr, "Unable to parse line \"$line\", ignoring...")
            continue
        end
    end

    return e2e_runtime, pseudo_transient_runtimes
end

function default_benchmark()
    exe = ARGS[2]
    report_path = exe * ".prj/reports"
    f, loop_latency = load_report_details(report_path)

    experiment_path = "experiments/max-res-default.json"
    experiment_data = JSON.parsefile(experiment_path)
    ly = experiment_data["ly"]
    lx = experiment_data["lx"]
    res = experiment_data["res"]

    out_path = Base.Filesystem.mkpath("./out/")
    command = `$exe $experiment_path $out_path`

    open(command, "r") do process_in
        _, pseudo_transient_runtimes = analyze_log(process_in)
        CSV.write("pseudo_transient_runtimes.csv", pseudo_transient_runtimes)

        best_performing_invocation = argmax(pseudo_transient_runtimes.pseudo_steps ./ pseudo_transient_runtimes.runtime)

        info = BenchmarkInformation(
            pseudo_transient_runtimes.pseudo_steps[best_performing_invocation],
            ly * res,
            lx * res,
            N_SUBITERATIONS,
            CELL_SIZE,
            OPERATIONS_PER_CELL,
            :monotile,
            N_CUS,
            TILE_HEIGHT,
            TILE_WIDTH,
            f,
            loop_latency,
            pseudo_transient_runtimes.runtime[best_performing_invocation]
        )

        metrics = Dict(
            "target" => "Convection",
            "n_cus" => N_CUS,
            "f" => f,
            "occupancy" => occupancy(info),
            "measured" => measured_throughput(info),
            "accuracy" => model_accurracy(info),
            "FLOPS" => measured_flops(info),
            "mem_throughput" => measured_mem_throughput(info)
        )

        open("metrics.json", "w") do metrics_file
            JSON.print(metrics_file, metrics)
        end
    end
end

mode = ARGS[1]
if mode == "default"
    default_benchmark()
end
