#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using DelimitedFiles
using Statistics

const GLOBAL_MEMORY_SPACE = 32 * 2^30
const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const TEMPORAL_PARALLELISM = Dict(:mono => 54, :multi_mono => 108, :tiling => 48, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:mono => 8, :multi_mono => 4, :tiling => 8, :cuda => 1)
const TILE_HEIGHT = Dict(:mono => 8192, :multi_mono => 4096, :tiling => 2^16, :cuda => nothing)
const TILE_WIDTH = Dict(:mono => 8192, :multi_mono => 4096, :tiling => 4096, :cuda => nothing)

function run_benchmark(exec, variant, n_ranks, grid_height, grid_width, n_iters; n_samples=5, run_warmup=true)
    experiment_dir = mktempdir("/dev/shm/")
    temp_path = experiment_dir * "/temp.bin"
    power_path = experiment_dir * "/power.bin"
    out_path = "/dev/null"

    run(`./data/input_gen.jl $grid_height $grid_width $temp_path $power_path`)

    command = `$exec $grid_height $grid_width $n_iters $temp_path $power_path $out_path`

    if variant == :multi_mono
        mpi_root = ENV["I_MPI_ROOT"]
        command = `$mpi_root/bin/mpirun -n $n_ranks $command`
    end

    if run_warmup
        if variant == :multi_mono
            warmup_cluster(command, n_ranks, variant)
        else
            run(command)
        end
    end

    runtimes = Vector()
    for _ in 1:n_samples
        runtime = open(command, "r") do process_in
            line_re = Regex("Walltime: ([0-9]+\\.[0-9]+) s")
            runtime = nothing

            while !eof(process_in)
                line = readline(process_in)
                println(line)

                line_match = match(line_re, line)
                if line_match !== nothing
                    runtime = parse(Float64, line_match[1])
                end
            end

            runtime
        end
        push!(runtimes, runtime)
    end
    runtime = minimum(runtimes)

    BenchmarkInformation(
        n_iters,
        grid_height,
        grid_width,
        n_ranks,
        1,
        CELL_SIZE,
        OPERATIONS_PER_CELL,
        variant,
        TEMPORAL_PARALLELISM[variant],
        SPATIAL_PARALLELISM[variant],
        TILE_HEIGHT[variant], # nothing in case of CUDA
        TILE_WIDTH[variant], # nothing in case of CUDA
        (variant == :cuda) ? nothing : load_report_details(exec * ".prj/reports"),
        runtime
    )
end

function max_perf_benchmark(exec, variant, n_ranks)
    if variant == :mono || variant == :multi_mono
        grid_height = TILE_HEIGHT[variant]
        grid_width = TILE_WIDTH[variant]
        n_iters = 2_000 * TEMPORAL_PARALLELISM[variant] * n_ranks
        n_samples = 5
    elseif variant == :tiling
        max_n_cells = GLOBAL_MEMORY_SPACE / 3 / CELL_SIZE
        n_tiles = Int(floor(√max_n_cells / TILE_WIDTH[:tiling]))
        @show grid_height = n_tiles * TILE_WIDTH[:tiling]
        grid_width = n_tiles * TILE_WIDTH[:tiling]
        n_iters = 100 * TEMPORAL_PARALLELISM[:tiling]
        n_samples = 5
    elseif variant == :cuda
        # TODO: Provide better parameters, maybe unify them.
        grid_height = grid_width = 16 * 2^10
        n_iters = 1000
        n_samples = 3
    end
    println("Grid dimensions: $(grid_height) x $(grid_width)")
    println("Grid size: $(grid_height * grid_width * CELL_SIZE * 2^-30) GB")
    println("No. of iterations: $(n_iters)")
    println("No. of samples: $(n_samples)")

    info = run_benchmark(exec, variant, n_ranks, grid_height, grid_width, n_iters; n_samples)

    if variant == :cuda
        target_name = "Hotspot, CUDA"
    elseif variant == :mono
        target_name = "Hotspot, Single-FPGA Monotile"
    elseif variant == :multi_mono
        target_name = "Hotspot, Multi-FPGA Monotile"
    elseif variant == :tiling
        target_name = "Hotspot, Tiling"
    end

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
    metrics
end

function deep_grid_scaling_benchmark(exec, variant, n_ranks)
    grid_wh = variant == :mono ? TILE_WIDTH[:mono] : max_grid_wh(variant, CELL_SIZE; clip_to_base=√2)

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
            continue
        end

        proto_info = BenchmarkInformation(
            # No. of iterations of one pass
            (variant == :cuda) ? 1 : n_ranks * TEMPORAL_PARALLELISM[variant],
            true_grid_wh,
            true_grid_wh,
            n_ranks,

            1, # No. of subiterations
            CELL_SIZE,
            OPERATIONS_PER_CELL,

            variant,
            TEMPORAL_PARALLELISM[variant],
            SPATIAL_PARALLELISM[variant],
            TILE_HEIGHT[variant],
            TILE_WIDTH[variant],

            (variant == :cuda) ? 1 : load_report_details(exec * ".prj/reports"), # Clock frequency

            42.0 # Dummy runtime
        )
        target_runtime = 30.0
        n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
        n_iters = n_passes * proto_info.n_iters

        info = run_benchmark(exec, variant, n_ranks, true_grid_wh, true_grid_wh, n_iters; n_samples=3, run_warmup=first_iteration)
        push!(df, [true_grid_wh, n_iters, info.runtime, measured_throughput(info), model_throughput(info)])
        CSV.write(df_path, df)

        grid_wh /= √2
        first_iteration = false
    end
end

if size(ARGS) != (4,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_ranks>")
    println(stderr, "Possible benchmarks: max_perf, strong_scaling, deep_grid_scaling")
    println(stderr, "Possible variants: mono, multi_mono, tiling, cuda")
    exit(1)
end

exec = ARGS[2]
variant = Symbol(ARGS[3])
n_ranks = parse(Int, ARGS[4])

if ARGS[1] == "max_perf"
    metrics = max_perf_benchmark(exec, variant, n_ranks)
    open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
elseif ARGS[1] == "deep_grid_scaling"
    deep_grid_scaling_benchmark(exec, variant, n_ranks)
elseif ARGS[1] == "strong_scaling"
    metrics = Dict()
    for i in n_ranks:-1:1
        metrics[i] = max_perf_benchmark(exec, variant, i)
        open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
    end
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
