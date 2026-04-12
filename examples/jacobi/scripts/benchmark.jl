#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using Glob

const MPI_ROOT = ENV["I_MPI_ROOT"]
const MPIRUN = "$MPI_ROOT/bin/mpirun"
const CELL_SIZE = 4 # bytes, including material coefficients

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

function run_benchmark(exe, n_ranks, config::JacobiConfig, grid_wh, n_timesteps; n_samples=3, run_warmup=true)
    arguments = fill(1/config.n_coefficients, config.n_coefficients)
    command = `$exe $(grid_wh) $(grid_wh) $(n_timesteps) /dev/null $(arguments)`

    # Set up the multi-FPGA cluster
    if config.variant == :multi_mono
        command = `$MPIRUN -n $n_ranks $command`
    end

    if run_warmup
        if config.variant == :multi_mono
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
    grid_wh = (config.variant == :mono || config.variant == :multi_mono) ? config.tile_width : max_grid_wh(config.variant, CELL_SIZE; clip_to_base=√2)

    proto_info = BenchmarkInformation(
        # No. of iterations of one pass
        (config.variant == :cuda) ? 1 : n_ranks * config.temporal_parallelism,
        grid_wh,
        grid_wh,
        n_ranks,

        1, # No. of subiterations
        CELL_SIZE,
        config.n_operations_per_cell,

        config.variant,
        config.temporal_parallelism,
        config.spatial_parallelism,
        config.tile_height,
        config.tile_width,

        (config.variant == :cuda) ? nothing : load_report_details(exe * ".prj/reports"), # Clock frequency

        42.0 # Dummy runtime
    )
    target_runtime = 30.0
    n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
    n_iters = n_passes * proto_info.n_iters

    info = run_benchmark(exe, n_ranks, config, grid_wh, n_iters)

    name_components = match(r"Jacobi([0-9])((Constant)|(General))", basename(exe))
    points = parse(Int, name_components[1])
    coef_type = name_components[2]
    if config.variant == :mono
        backend = "Single-FPGA Monotile"
    elseif config.variant == :multi_mono
        backend = "Multi-FPGA Monotile"
    elseif config.variant == :tiling
        backend = "Tiling"
    elseif config.variant == :cuda
        backend = "CUDA"
    end
    target_name = "$(points)-point $coef_type Jacobi, $backend"

    if config.variant == :cuda
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

    open("metrics.$(basename(exe)).json", "w") do metrics_file
        JSON.print(metrics_file, metrics)
    end
end

function deep_grid_scaling_benchmark(exe, n_ranks, config::JacobiConfig)
    grid_wh = config.variant == :mono ? config.tile_width : max_grid_wh(config.variant, CELL_SIZE; clip_to_base=√2)

    df_path = "scaling.$(config.variant).csv"
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
            continue
        end

        proto_info = BenchmarkInformation(
            # No. of iterations of one pass
            (config.variant == :cuda) ? 1 : n_ranks * config.temporal_parallelism,
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

            (config.variant == :cuda) ? nothing : load_report_details(exe * ".prj/reports"), # Clock frequency

            42.0 # Dummy runtime
        )
        target_runtime = 30.0
        n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
        n_iters = n_passes * proto_info.n_iters

        info = run_benchmark(exe, n_ranks, config, true_grid_wh, n_iters; n_samples=3, run_warmup=first_iteration)
        push!(df, [true_grid_wh, n_iters, info.runtime, measured_throughput(info), model_throughput(info)])
        CSV.write(df_path, df)

        grid_wh /= √2
        first_iteration = false
    end
end

function deep_grid_scaling_ncu_profile(exe, config::JacobiConfig)
    grid_wh = max_grid_wh(:cuda, CELL_SIZE; clip_to_base=2)
    arguments = fill(1/config.n_coefficients, config.n_coefficients)

    df_path = "scaling.ncu.$(config.variant).csv"
    df = nothing

    while round(grid_wh) >= 32
        true_grid_wh = Int(round(grid_wh))

        command = `$exe $true_grid_wh $true_grid_wh 1 /dev/null $arguments`

        data = ncu_profile_command(command; launch_id=0)
        data[:grid_wh] = true_grid_wh

        if df === nothing
            df = DataFrame(data)
        else
            push!(df, data)
        end
        CSV.write(df_path, df)

        grid_wh /= √2
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
elseif ARGS[1] == "deep_grid_scaling_ncu"
    deep_grid_scaling_ncu_profile(exe, config)
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
