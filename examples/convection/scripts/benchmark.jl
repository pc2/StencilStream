#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")

const N_SUBITERATIONS = 3
const OPERATIONS_PER_CELL = (5 + 5 + 3 + 6 + 6 + 6) + (10 + 3 + 2 + 14 + 3 + 2) + 2
const CELL_SIZE = 88 # bytes
const TEMPORAL_PARALLELISM = Dict(:mono => 6, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:mono => 1, :cuda => 1)
const TILE_NX = Dict(:mono => 2^16, :cuda => nothing)
const TILE_NY = Dict(:mono => 512, :cuda => nothing)

function run_benchmark(exe, variant, experiment)
    (experiment_path, experiment_file) = mktemp()
    JSON.print(experiment_file, experiment)
    flush(experiment_file)

    out_path = Base.Filesystem.mkpath("./out/")
    command = `$exe $experiment_path $out_path`

    computation_time, n_iterations = open(command) do logfile
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

        computation_time, n_iterations
    end

    lx = experiment["lx"]
    ly = experiment["ly"]
    res = experiment["res"]
    BenchmarkInformation(
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
end

function max_perf_benchmark(exe, variant)
    if variant == :cuda
        experiment_path = "experiments/cuda-benchmark.json"
    else
        experiment_path = "experiments/max-res-default.json"
    end
    experiment = JSON.parsefile(experiment_path)
    
    info = run_benchmark(exe, variant, experiment)
    
    if variant == :cuda
        metrics = Dict(
            "target" => "Convection, CUDA",
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        metrics = Dict(
            "target" => "Convection, Single-FPGA Monotile",
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

function deep_grid_scaling_benchmark(exe, variant)
    grid_wh = variant == :mono ? TILE_NY[:mono] : max_grid_wh(variant, CELL_SIZE; clip_to_base=2)

    df_path = "scaling.$variant.csv"
    if isfile(df_path)
        df = CSV.read(df_path, DataFrame)
    else
        df = DataFrame(grid_wh=Int64[], n_iters=Int64[], runtime=Float64[], measured_throughput=Float64[], model_throughput=Float64[])
    end

    first_iteration = true
    while round(grid_wh) >= 32
        true_grid_wh = Int(round(grid_wh))

        if true_grid_wh ∈ df.grid_wh
            grid_wh /= √2
        end

        proto_info = BenchmarkInformation(
            (variant == :cuda) ? 1 : n_ranks * TEMPORAL_PARALLELISM[variant],
            true_grid_wh,
            true_grid_wh,
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

            42.0 # Dummy runtime
        )
        target_runtime = 30.0
        n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
        n_iters = n_passes * proto_info.n_iters

        experiment = Dict(
            "ly" => 1.0,
            "lx" => 1.0,
            "py" => 1.0,
            "px" => 1.5,
            "res" => true_grid_wh,

            "eta0" => 1.0,
            "DcT" => 1.0,
            "deltaT" => 1.0,
            "Ra" => 1e7,
            "Pra" => 1e3,

            "iterMax" => n_iters,
            "nt" => 1,
            "nout" => 1,
            "nerr" => n_iters,
            "epsilon" => 1e-4,
            "dmp" => 2
        )

        if first_iteration
            # Warmup
            run_benchmark(exe, variant, experiment)
            first_iteration = false
        end
        infos = [run_benchmark(exe, variant, experiment) for _ in 1:3]
        info = argmin(i -> i.runtime, infos)
        push!(df, [true_grid_wh, n_iters, info.runtime, measured_throughput(info), model_throughput(info)])
        CSV.write(df_path, df)

        grid_wh /= √2
    end
end

function print_usage()
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_fpgas>")
    println(stderr, "Possible benchmarks: max_perf, deep_grid_scaling")
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
elseif mode == "deep_grid_scaling"
    deep_grid_scaling_benchmark(exe, variant)
end
