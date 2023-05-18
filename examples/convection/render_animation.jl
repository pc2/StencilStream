using DelimitedFiles
using Statistics
using CairoMakie

usage_message = "Usage: " * PROGRAM_FILE * " <output directory> [<reference_output directory>]"

if size(ARGS,1) < 1
    println(stderr, usage_message)
    exit(1)
end

output_directory_path = ARGS[1]
if !isdir(output_directory_path)
    println("Output directory "* output_directory_path * "does not exist or is not a directory!")
    exit(1)
end
temp_indices = [parse(Int, first(eachsplit(filename, "."))) for filename in readdir(output_directory_path)]
sort!(temp_indices)
temp_files = (output_directory_path * "/") .* string.(temp_indices) .* ".csv"

if size(ARGS,1) >= 2
    reference_output_directory = ARGS[2]
    if !isdir(reference_output_directory)
        println("Reference output directory " * output_directory_path * "does not exist or is not a directory!")
        exit(1)
    end
else
    reference_output_directory = nothing
end

function load_file(file_path)
    readdlm(open(file_path), ',', Float64, '\n')'
end

println("Reading files")
temps = Vector{Matrix}(undef, size(temp_files))
diffs = Vector{Matrix}(undef, size(temp_files))
Threads.@threads for i in eachindex(temp_files)
    temp_file_path = temp_files[i]
    temps[i] = load_file(temp_file_path)
    if reference_output_directory !== nothing
        reference_file_path = reference_output_directory * "/" * basename(temp_file_path)
        reference_temp = load_file(reference_file_path)
        diffs[i] = abs.(temps[i] .- reference_temp)
    end
end
diffs

println("Visualizing")
index = Observable(1)
if reference_output_directory === nothing
    figure_height = 500
else
    figure_height = 1000
end
fig = Figure(resolution=(1500, figure_height))

temp_map = @lift(temps[$index])
min_temp = minimum(minimum.(temps))
max_temp = maximum(maximum.(temps))

Axis(fig[1,1], title="Simulated temperature", aspect=DataAspect())
heatmap!(fig[1,1], temp_map, interpolate=true; colorrange=(min_temp, max_temp), colormap=:inferno)
Colorbar(fig[1,2], limits=(min_temp, max_temp), colormap=:inferno, label="T°")

if reference_output_directory !== nothing
    diff_map = @lift(diffs[$index])
    max_diff = maximum(maximum.(diffs))

    Axis(fig[2,1], title="Difference to reference", aspect=DataAspect())
    heatmap!(fig[2,1], diff_map, interpolate=true; colorrange=(0, max_diff), colormap=:inferno)
    Colorbar(fig[2,2], limits=(0, max_diff), colormap=:inferno, label="T°")
end

record(fig, "animation.mp4", eachindex(diffs); framereate=15) do i
    index[] = i
end