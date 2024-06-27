using DataFrames
using CSV
using Statistics
using JSON
using CairoMakie

function parse_report_js(path)
    fields = Dict{String,Any}()
    for line in readlines(path)
        if occursin("fileJSON", line)
            continue # Avoiding to load all source files. They're too big for REs.
        end

        matched_line = match(r"var (.+)=(\{.+\});$", line)
        if matched_line === nothing
            continue
        end

        name = matched_line[1]
        data = replace(matched_line[2], "\\'" => "'")
        fields[name] = JSON.parse(data)
    end
    return fields
end

function load_report_details(report_path)
    if !isdir(report_path)
        println(stderr, "Could not find synthesis report, using stand-in numbers for testing")
        return 350e6, 2000
    end

    quartus_data = parse_report_js("$report_path/resources/quartus_data.js")
    f = parse(Float32, quartus_data["quartusJSON"]["quartusFitClockSummary"]["nodes"][1]["kernel clock fmax"]) * 1e6

    report_data = parse_report_js("$report_path/resources/report_data.js")
    kernels = report_data["loop_attrJSON"]["nodes"]
    loops = Iterators.flatmap(kernel -> kernel["children"], kernels)
    loop_latency = sum(Iterators.map(loop -> parse(Float32, loop["lt"]), loops))

    return f, loop_latency
end

struct BenchmarkInformation
    # Dynamic benchmark parameters
    n_iters::Int
    n_grid_rows::Int
    n_grid_cols::Int

    # Static/code parameters
    n_subiters::Int
    cell_size::Int
    ops_per_cell::Int

    # Implementation parameters
    variant::Symbol
    n_cus::Int
    n_tile_rows::Int
    n_tile_cols::Int

    # Synthesis results
    f::Float64
    loop_latency::Int

    # Benchmark results
    runtime::Float64
end

grid_size(info::BenchmarkInformation) = info.n_grid_rows * info.n_grid_cols
workload(info::BenchmarkInformation) = grid_size(info) * info.n_iters
n_passes(info::BenchmarkInformation) = ceil(info.n_iters * info.n_subiters / info.n_cus)
padded_cell_size(info::BenchmarkInformation) = 2^ceil(log2(info.cell_size))
max_cell_rate(info::BenchmarkInformation) = ceil(padded_cell_size(info) / 64)

function transfered_cells(info::BenchmarkInformation)
    if info.variant == :monotile
        return n_passes(info) * 2grid_size(info)
    elseif info.variant == :tiling
        transfered_cells = 0
        for tile_start_r in 1:info.n_tile_rows:info.n_grid_rows
            for tile_start_c in 1:info.n_tile_cols:info.n_grid_cols
                output_tile_height = min(info.n_grid_rows, tile_start_r + info.n_tile_rows) - tile_start_r + 1
                output_tile_width = min(info.n_grid_cols, tile_start_c + info.n_tile_cols) - tile_start_c + 1
                transfered_cells +=
                    (2info.n_cus + output_tile_height) * (2info.n_cus + output_tile_width) +
                    output_tile_height * output_tile_width
            end
        end
        return n_passes(info) * transfered_cells
    else
        throw(KeyError(info.variant))
    end
end

measured_throughput(info::BenchmarkInformation) = workload(info) / info.runtime
measured_mem_throughput(info::BenchmarkInformation) = transfered_cells(info) * padded_cell_size(info) / info.runtime
measured_flops(info::BenchmarkInformation) = measured_throughput(info) * info.ops_per_cell

function model_runtime(info::BenchmarkInformation)
    if info.variant == :monotile
        cu_latency = info.n_grid_rows + 1
        pipeline_latency = info.n_cus * cu_latency
        n_loop_iterations = pipeline_latency + (info.n_grid_rows * info.n_grid_cols)
        n_cycles_per_pass = n_loop_iterations + info.loop_latency
    elseif info.variant == :tiling
        n_cycles_per_pass = 0
        for tile_col in 1:ceil(info.n_grid_cols / info.n_tile_cols)
            for tile_row in 1:ceil(info.n_grid_rows / info.n_tile_rows)
                tile_section_width = min(info.n_tile_cols, info.n_grid_cols - (tile_col - 1) * info.n_tile_cols)
                tile_section_height = min(info.n_tile_rows, info.n_grid_rows - (tile_row - 1) * info.n_tile_rows)
                n_loop_iterations_per_tile = (tile_section_width + 2info.n_cus) * (tile_section_height + 2info.n_cus)
                n_cycles_per_pass += info.loop_latency + n_loop_iterations_per_tile
            end
        end
    else
        throw(KeyError(info.variant))
    end

    return n_passes(info) * max_cell_rate(info) * n_cycles_per_pass / info.f
end
model_throughput(info::BenchmarkInformation) = workload(info) / model_runtime(info)
max_execute_throughput(info::BenchmarkInformation) = info.n_cus / info.n_subiters * info.f

model_accurracy(info::BenchmarkInformation) = model_throughput(info) / measured_throughput(info)
occupancy(info::BenchmarkInformation) = measured_throughput(info) / max_execute_throughput(info)

function render_model_error(df, file_name)
    fig = Figure(size=(1600, 600))
    azimuth = Observable(0.0)
    ax = Axis3(fig[1, 1], xlabel="no. of timesteps", ylabel="grid width/height", zlabel="", title="relative model error [percent]", azimuth=azimuth)

    new_df = DataFrame(n_timesteps=Int64[], grid_wh=Int64[], kernel_runtime=Float64[], model_runtime=Float64[])

    for n_timesteps in Set(df.n_timesteps)
        for grid_wh in Set(df.grid_wh)
            sub_df = df[df.n_timesteps.==n_timesteps.&&df.grid_wh.==grid_wh, [:kernel_runtime, :model_runtime]]
            kernel_runtime = mean(sub_df.kernel_runtime)
            model_runtime = mean(sub_df.model_runtime)
            push!(new_df, [n_timesteps, grid_wh, kernel_runtime, model_runtime])
        end
    end

    surface!(ax, new_df.n_timesteps, new_df.grid_wh, new_df.kernel_runtime ./ new_df.model_runtime .* 100.0)

    record(fig, file_name, LinRange(0.0, 2.0π, 240); framerate=24) do a
        azimuth[] = a
    end
end
