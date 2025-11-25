#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using DelimitedFiles
using Statistics

const GLOBAL_MEMORY_SPACE = 32 * 2^30
const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const TEMPORAL_PARALLELISM = Dict(:monotile => 108, :tiling => 48, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:monotile => 4, :tiling => 8, :cuda => 1)
const TILE_HEIGHT = Dict(:monotile => 4096, :tiling => 2^16, :cuda => nothing)
const TILE_WIDTH = Dict(:monotile => 4096, :tiling => 4096, :cuda => nothing)

function run_benchmark(exec, variant, n_ranks, grid_height, grid_width, n_iters; n_samples=5, warmup_time="2m", run_warmup=true)
    experiment_dir = mktempdir("/dev/shm/")
    temp_path = experiment_dir * "/temp.bin"
    power_path = experiment_dir * "/power.bin"
    out_path = "/dev/null"

    run(`./data/input_gen.jl $grid_height $grid_width $temp_path $power_path`)

    command = `$exec $grid_height $grid_width $n_iters $temp_path $power_path $out_path`
    if variant == :monotile
        mpi_root = ENV["I_MPI_ROOT"]
        command = `$mpi_root/bin/mpirun -n $n_ranks $command`
    end

    if run_warmup
        if variant == :monotile
            warmup_cluster(command, n_ranks, variant; warmup_time)
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
    if variant == :monotile
        grid_height = TILE_HEIGHT[:monotile]
        grid_width = TILE_WIDTH[:monotile]
        n_iters = 2_000 * TEMPORAL_PARALLELISM[:monotile]
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
        metrics = Dict(
            "target" => "Hotspot, CUDA",
            "measured" => measured_throughput(info),
            "FLOPS" => measured_flops(info)
        )
    else
        metrics = Dict(
            "target" => (variant == :monotile) ? "Hotspot, Monotile" : "Hotspot, Tiling",
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
    if variant == :monotile
        grid_wh = 4096 # Maximum the design can support
    elseif variant == :tiling
        grid_wh = 32768 # Maximum would be 36864, 32768 is next-lowest power of two
    elseif variant == :cuda
        grid_wh = 32768 # Maximum, given 40 GB global memory
    end

    df_path = "scaling.$variant.csv"
    df = DataFrame(grid_wh=Int64[], n_iters=Int64[], runtime=Float64[], measured_throughput=Float64[])

    first_iteration = true
    while grid_wh^2 >= 32
        true_grid_wh = Int(ceil(grid_wh))
        target_runtime = 30.0
        if variant == :monotile || variant == :tiling
            proto_info = BenchmarkInformation(
                n_ranks * TEMPORAL_PARALLELISM[variant], # Dummy number of iterations for one pass
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

                load_report_details(exec * ".prj/reports"), # Clock frequency

                42.0 # Dummy runtime
            )
            n_passes = Int(ceil(target_runtime / model_runtime(proto_info)))
            n_iters = n_ranks * TEMPORAL_PARALLELISM[variant] * n_passes
        else
            mem_throughput = 1555.0 * 2^30
            cell_rate = mem_throughput / 2CELL_SIZE
            iteration_rate = cell_rate / true_grid_wh^2
            n_iters = Int(ceil(iteration_rate * target_runtime))
        end

        info = run_benchmark(exec, variant, n_ranks, true_grid_wh, true_grid_wh, n_iters; n_samples=3, run_warmup=first_iteration)
        push!(df, [true_grid_wh, n_iters, info.runtime, measured_throughput(info)])
        CSV.write(df_path, df)

        grid_wh /= √2
        first_iteration = false
    end
end

if size(ARGS) != (4,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_ranks>")
    println(stderr, "Possible benchmarks: max_perf, strong_scaling, deep_grid_scaling")
    println(stderr, "Possible variants: monotile, tiling, cuda")
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
