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

function model_monotile_runtime(f, loop_latency, n_grid_rows, n_grid_cols, n_gens, n_cus)
    cu_latency = n_grid_rows + 1
    pipeline_latency = n_cus * cu_latency
    n_iterations = pipeline_latency + (n_grid_rows * n_grid_cols)
    n_cycles = n_iterations + loop_latency
    
    ceil(n_gens / n_cus) * n_cycles / f
end

function model_tiling_runtime(f, loop_latency, n_grid_rows, n_grid_cols, n_gens, n_tile_rows, n_tile_cols, n_cus)
    n_cycles_per_pass = 0
    for tile_col in 1:ceil(n_grid_cols / n_tile_cols)
        n_tiles_in_column = ceil(n_grid_rows / n_tile_rows)
        tile_section_width = min(n_tile_cols, n_grid_cols - (tile_col-1) * n_tile_cols)
        n_cycles_per_pass += n_tiles_in_column * (loop_latency + (tile_section_width + 2n_cus) * (n_tile_rows + 2n_cus))
    end

    ceil(n_gens / n_cus) * n_cycles_per_pass / f
end

function render_model_error(df, file_name)
    fig = Figure(size=(1600, 600))
    azimuth = Observable(0.0)
    ax = Axis3(fig[1,1], xlabel="no. of timesteps", ylabel="grid width/height", zlabel="", title="relative model error [percent]", azimuth=azimuth)

    new_df = DataFrame(n_timesteps=Int64[], grid_wh=Int64[], kernel_runtime=Float64[], model_runtime=Float64[])

    for n_timesteps in Set(df.n_timesteps)
        for grid_wh in Set(df.grid_wh)
            sub_df = df[df.n_timesteps .== n_timesteps .&& df.grid_wh .== grid_wh, [:kernel_runtime, :model_runtime]]
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

function build_metrics(measured_runtime, n_gens, variant, f, loop_latency, n_grid_rows, n_grid_cols, n_tile_rows, n_tile_cols, n_cus, ops_per_cell, cell_size)
    n_passes = ceil(n_gens / n_cus)
    grid_size = n_grid_cols * n_grid_rows
    workload = n_gens * grid_size
    measured_rate = workload / measured_runtime

    if variant == :monotile
        model_runtime = model_monotile_runtime(f, loop_latency, n_grid_rows, n_grid_cols, n_gens, n_cus)
        transfered_cells = 2 * grid_size * n_passes
    else
        model_runtime = model_tiling_runtime(f, loop_latency, n_grid_rows, n_grid_cols, n_gens, n_tile_rows, n_tile_cols, n_cus)
        transfered_cells = ((2n_cus + n_grid_rows)*(2n_cus + n_grid_cols) + grid_size) * n_passes
    end
    model_rate = workload / model_runtime
    max_rate = f * n_cus

    Dict(
        :measured_rate => measured_rate,
        :model_rate => model_rate,
        :max_rate => max_rate,
        :flops => measured_rate * ops_per_cell,
        :mem_throughput => transfered_cells * cell_size / measured_runtime,
        :occupancy => measured_rate / max_rate,
        :model_accurracy => model_runtime / measured_runtime,
    )
end
