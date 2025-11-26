using DataFrames
using CSV
using Statistics
using JSON

function parse_report_js(path)
    fields = Dict{String,Any}()
    for line in readlines(path)
        if occursin("fileNDJSON", line) || occursin("fileNDJSON", line)
            continue # Avoiding to load all source files. They're too big for REs.
        end

        matched_line = match(r"var (.+)=(.+);$", line)
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
        return 350e6
    end

    quartus_data = parse_report_js("$report_path/resources/quartus_data.js")
    # Checking whether this is an >=25.0.0 report.
    if "quartusNDJSON" ∈ keys(quartus_data)
        parse(Float32, quartus_data["quartusNDJSON"][1]["quartusFitClockSummary"]["nodes"][1]["kernel clock"]) * 1e6
    else
        parse(Float32, quartus_data["quartusJSON"]["quartusFitClockSummary"]["nodes"][1]["kernel clock"]) * 1e6
    end
end

struct BenchmarkInformation
    # Dynamic benchmark parameters
    n_iters::Int
    n_grid_rows::Int
    n_grid_cols::Int
    n_ranks::Union{Int, Nothing}

    # Static/code parameters
    n_subiters::Int
    cell_size::Int
    ops_per_cell::Int

    # Implementation parameters
    variant::Symbol
    temporal_parallelism::Union{Int,Nothing} # may be nothing for CUDA variant
    spatial_parallelism::Union{Int,Nothing} # may be nothing for CUDA variant
    n_tile_rows::Union{Int,Nothing} # may be nothing for CUDA variant
    n_tile_cols::Union{Int,Nothing} # may be nothing for CUDA variant

    # Synthesis results
    f::Union{Float64,Nothing} # may be nothing for CUDA variant

    # Benchmark results
    runtime::Float64
end

function f_effective(info::BenchmarkInformation)
    padded_vector_size =  2^ceil(log2(info.cell_size * info.spatial_parallelism))
    if !isnothing(info.n_ranks) && info.variant == :multi_mono
        s_link = 32 # pipeword size
        f_link_single = 5.0e9 / s_link # clock rate of single IO pipe
        f_link = 2f_link_single * s_link / padded_vector_size
    else
        f_link = Inf
    end
    mem_width = 64
    f_mem = info.f * mem_width / padded_vector_size
    minimum([f_link, f_mem, info.f])
end

grid_size(info::BenchmarkInformation) = info.n_grid_rows * info.n_grid_cols
workload(info::BenchmarkInformation) = grid_size(info) * info.n_iters
function n_passes(info::BenchmarkInformation)
    if info.variant == :cuda
        nothing
    elseif (info.variant == :multi_mono && !isnothing(info.n_ranks))
        ceil(info.n_iters / info.temporal_parallelism / info.n_ranks)
    else
        ceil(info.n_iters / info.temporal_parallelism)
    end
end
n_cus(info::BenchmarkInformation) = info.variant == :cuda ? nothing : info.temporal_parallelism * info.n_subiters
function parallelity(info::BenchmarkInformation)
    if info.variant == :cuda
        nothing
    else
        (isnothing(info.n_ranks) ? 1 : info.n_ranks) * info.temporal_parallelism * info.spatial_parallelism
    end
end

halo_height(info::BenchmarkInformation) = (info.variant == :tiling) ? n_cus(info) : nothing
halo_width(info::BenchmarkInformation) = (info.variant == :tiling) ? info.spatial_parallelism * n_cus(info) : nothing
n_grid_col_vects(info::BenchmarkInformation) = ceil(info.n_grid_cols / info.spatial_parallelism)
n_tile_col_vects(info::BenchmarkInformation) = info.n_tile_cols / info.spatial_parallelism

measured_throughput(info::BenchmarkInformation) = workload(info) / info.runtime
measured_flops(info::BenchmarkInformation) = measured_throughput(info) * info.ops_per_cell

function model_runtime(info::BenchmarkInformation)
    l_link = 0.5e-3 / info.f

    if info.variant == :mono || info.variant == :multi_mono
        l_compute_unit = n_grid_col_vects(info) + 1
        l_fpga = n_cus(info) * l_compute_unit

        c_prime = l_fpga + (info.n_grid_rows * n_grid_col_vects(info))
        c_pass = c_prime
        if !isnothing(info.n_ranks)
            c_pass += (info.n_ranks - 1) * (l_fpga + l_link)
        end
    elseif info.variant == :tiling
        c_pass = 0
        for tile_row in 1:ceil(info.n_grid_rows / info.n_tile_rows)
            for tile_col in 1:ceil(info.n_grid_cols / info.n_tile_cols)
                tile_section_height = min(info.n_tile_rows, info.n_grid_rows - (tile_row - 1) * info.n_tile_rows)
                tile_section_width = min(n_tile_col_vects(info), n_grid_col_vects(info) - (tile_col - 1) * n_tile_col_vects(info))
                n_loop_iterations_per_tile = (tile_section_width + 2n_cus(info)) * (tile_section_height + 2n_cus(info))
                c_pass += n_loop_iterations_per_tile
            end
        end
    elseif info.variant == :cuda
        throw("not implemented")
    else
        throw(KeyError(info.variant))
    end

    return n_passes(info) * c_pass / f_effective(info)
end
model_throughput(info::BenchmarkInformation) = workload(info) / model_runtime(info)
max_compute_throughput(info::BenchmarkInformation) = parallelity(info) * info.f

model_accurracy(info::BenchmarkInformation) = model_throughput(info) / measured_throughput(info)
occupancy(info::BenchmarkInformation) = measured_throughput(info) / max_compute_throughput(info)

function setup_io_pipes(n_ranks, variant)
    if variant != :multi_mono
        throw("not implemented")
    end
    command = ["changeFPGAlinks"]
    for i_rank in 0:(n_ranks-2)
        i_node = i_rank ÷ 2
        i_acl = i_rank % 2
        i_next_node = (i_rank + 1) ÷ 2
        i_next_acl = (i_rank + 1) % 2

        push!(command, "--fpgalink=n$(lpad(i_node,2,'0')):acl$(i_acl):ch2-n$(lpad(i_next_node,2,'0')):acl$(i_next_acl):ch0")
        push!(command, "--fpgalink=n$(lpad(i_node,2,'0')):acl$(i_acl):ch3-n$(lpad(i_next_node,2,'0')):acl$(i_next_acl):ch1")
    end
    i_last_node = (n_ranks - 1) ÷ 2
    i_last_acl = (n_ranks - 1) % 2
    push!(command, "--fpgalink=n$(lpad(i_last_node,2,'0')):acl$(i_last_acl):ch2-n00:acl0:ch0")
    push!(command, "--fpgalink=n$(lpad(i_last_node,2,'0')):acl$(i_last_acl):ch3-n00:acl0:ch1")
    command = Cmd(command)
    run(command)
end

function warmup_cluster(command, n_ranks, variant; links_preconfigured=false)
    warmup_successful = false
    for warmup_try in 1:3
        if warmup_try > 1 || !links_preconfigured
            setup_io_pipes(n_ranks, variant)
        end
        r = run(Cmd(`timeout 2m $command`, ignorestatus=true))
        if r.exitcode == 0
            warmup_successful = true
            break
        end
    end
    if !warmup_successful
        println(stderr, "Failed to warmup links!")
        exit(1)
    end
end
