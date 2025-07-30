#!/usr/bin/env -S julia --project
include("../../../scripts/benchmark-common.jl")
using DelimitedFiles
using Statistics

const GLOBAL_MEMORY_SPACE = 32 * 2^30
const OPERATIONS_PER_CELL = 15
const CELL_SIZE = 8 # bytes
const TEMPORAL_PARALLELISM = Dict(:monotile => 128, :tiling => 64, :cuda => 1)
const SPATIAL_PARALLELISM = Dict(:monotile => 4, :tiling => 8, :cuda => 1)
const TILE_HEIGHT = Dict(:monotile => 4096, :tiling => 2^16, :cuda => nothing)
const TILE_WIDTH = Dict(:monotile => 4096, :tiling => 4096, :cuda => nothing)

function create_experiment(n_rows, n_columns, temp_file, power_file)
    begin
        temp = fill(30.0f0, n_rows * n_columns)
        write(temp_file, temp)
    end

    begin
        power = zeros(Float32, n_rows, n_columns)
        power[(n_rows÷4):(3n_rows÷4), (n_columns÷4):(3n_columns÷4)] .= 0.5
        power = reshape(power', n_rows * n_columns)
        write(power_file, power)
    end
end

function max_perf_benchmark(exec, variant, n_ranks)
    setup_io_pipes(n_ranks, variant)

    if variant == :monotile
        grid_height = TILE_HEIGHT[:monotile]
        grid_width = TILE_WIDTH[:monotile]
        n_iters = 2_000 * TEMPORAL_PARALLELISM[:monotile]
        n_samples = 5
    elseif variant == :tiling
        max_n_cells = GLOBAL_MEMORY_SPACE / 3 / CELL_SIZE
        n_tiles = Int(floor(√max_n_cells / TILE_WIDTH[:tiling]))
        grid_height = n_tiles * TILE_WIDTH[:tiling]
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

    experiment_dir = mktempdir("/dev/shm/")
    temp_path = experiment_dir * "/temp.bin"
    power_path = experiment_dir * "/power.bin"
    out_path = "/dev/null"
    println("temp: $temp_path, power: $power_path")

    println("Creating experiment...")
    create_experiment(grid_height, grid_width, temp_path, power_path)
    println("Experiment created and written!")

    mpi_root = ENV["I_MPI_ROOT"]
    command = `timeout 3m $mpi_root/bin/mpirun -n $n_ranks $exec $grid_height $grid_width $n_iters $temp_path $power_path $out_path`

    # Warmup to exclude programming from the benchmark and to test for functionality
    r = run(Cmd(command, ignorestatus=true))
    if r.exitcode == 124
        return nothing
    end

    runtimes = Vector()
    for i_sample in 1:n_samples
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

    info = BenchmarkInformation(
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

if size(ARGS) != (4,)
    println(stderr, "Usage: $PROGRAM_FILE <benchmark> <path to executable> <variant> <n_ranks>")
    println(stderr, "Possible benchmarks: max_perf")
    println(stderr, "Possible variants: monotile, tiling, cuda")
    exit(1)
end

exec = ARGS[2]
variant = Symbol(ARGS[3])
n_ranks = parse(Int, ARGS[4])

if ARGS[1] == "max_perf"
    metrics = max_perf_benchmark(exec, variant, n_ranks)
    open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
elseif ARGS[1] == "strong_scaling"
    metrics = Dict()
    for i in n_ranks:-1:1
        try
            metrics[i] = max_perf_benchmark(exec, variant, i)
        catch e
            if isa(e, ProcessFailedException)

            else
                throw(e)
            end
        end
        open(f -> JSON.print(f, metrics), "metrics.$variant.json", "w")
    end
else
    println(stderr, "Unknown benchmark '$(ARGS[1])'")
    exit(1)
end
