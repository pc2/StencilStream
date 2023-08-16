using DataFrames
using CSV
using Statistics
using JSON

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

function model_monotile_throughput(f, loop_latency, n_grid_rows, n_grid_cols, n_cus)
    cu_latency = n_grid_rows + 1
    pipeline_latency = n_cus * cu_latency
    n_iterations = pipeline_latency + (n_grid_rows * n_grid_cols)
    n_cycles = n_iterations + loop_latency
    work = (n_grid_rows * n_grid_cols) * n_cus
    work / n_cycles * f
end

function model_tiling_throughput(f, loop_latency, n_grid_rows, n_grid_cols, n_tile_rows, n_tile_cols, n_cus)

    n_cycles = 0
    for tile_col in 1:ceil(n_grid_cols / n_tile_cols)
        for tile_row in 1:ceil(n_grid_rows / n_tile_rows)
            tile_section_width = min(n_tile_cols, n_grid_cols - (tile_col-1) * n_tile_cols)
            tile_section_height = min(n_tile_rows, n_grid_rows - (tile_row-1) * n_tile_rows)
            n_cycles += loop_latency + (tile_section_width + 2n_cus) * (tile_section_height + 2n_cus)
        end
    end

    work = (n_grid_rows * n_grid_cols) * n_cus
    work / n_cycles * f
end

function build_metrics(target, measured_runtime, n_gens, variant, f, loop_latency, n_grid_rows, n_grid_cols, n_tile_rows, n_tile_cols, n_cus, ops_per_cell, cell_size)
    n_passes = ceil(n_gens / n_cus)

    if variant == :monotile
        cu_latency = n_grid_rows + 1
        pipeline_latency = n_cus * cu_latency
        n_loop_iterations_per_pass = pipeline_latency + (n_grid_rows * n_grid_cols)
        n_cycles_per_pass = n_loop_iterations_per_pass + loop_latency
        n_cycles = n_cycles_per_pass * n_passes

        transfered_cells = 2 * n_grid_cols * n_grid_rows * n_passes
    elseif variant == :tiling
        n_cycles_per_pass = 0
        transfered_cells_per_pass = 0
        for tile_col in 1:ceil(n_grid_cols / n_tile_cols)
            for tile_row in 1:ceil(n_grid_rows / n_tile_rows)
                tile_section_width = min(n_tile_cols, n_grid_cols - (tile_col-1) * n_tile_cols)
                tile_section_height = min(n_tile_rows, n_grid_rows - (tile_row-1) * n_tile_rows)

                n_cycles_per_pass += loop_latency + (tile_section_width + 2n_cus) * (tile_section_height + 2n_cus)

                center_cells = tile_section_width * tile_section_height
                halo_cells = 2(tile_section_width * n_cus) + 2(tile_section_height * n_cus) + 4n_cus^2
                transfered_cells_per_pass += 2center_cells + halo_cells
            end
        end

        n_cycles = n_cycles_per_pass * n_passes
        transfered_cells = transfered_cells_per_pass * n_passes
    end

    n_updates = n_grid_rows * n_grid_cols * n_gens
    model_runtime = n_cycles / f

    measured_performance = n_updates / measured_runtime
    model_performance = n_updates / model_runtime

    max_peak_performance = f * n_cus
    occupancy = measured_performance / max_peak_performance
    model_accurracy = measured_performance / model_performance

    flops = ops_per_cell * measured_performance
    mem_throughput = transfered_cells * cell_size / measured_runtime

    return Dict(
        "target" => target,
        "n_cus" => n_cus,
        "f" => f,
        "occupancy" => occupancy,
        "measured" => measured_performance,
        "accuracy" => model_accurracy,
        "FLOPS" => flops,
        "mem_throughput" => mem_throughput,
    )
end