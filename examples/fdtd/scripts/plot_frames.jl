#!/usr/bin/env -S julia --project=../..
using DelimitedFiles
using Statistics
using CairoMakie
using Glob

if size(ARGS,1) < 2
    println(stderr, "Usage: " * PROGRAM_FILE * " <target> <output directory>")
    exit(1)
end

target = Symbol(ARGS[1])
if target ∉ [:frames, :animation]
    println(stderr, "Unknown target $target!")
    exit(1)
end

output_directory_path = ARGS[2]
if !isdir(output_directory_path)
    println("Output directory "* output_directory_path * " does not exist or is not a directory!")
    exit(1)
end

hz_file_paths = glob("hz.*.csv", output_directory_path)
hz_indices = parse.(Int, getindex.(match.(r"hz\.([0-9]+)\.csv", hz_file_paths), 1))
sort!(hz_indices)
hz_file_paths = (output_directory_path * "/hz.") .* string.(hz_indices) .* ".csv"

if (target == :frames)
    hz_sum_file_paths = glob("hz_sum.*.csv", output_directory_path)
    append!(hz_file_paths, hz_sum_file_paths)
end

hzs = Vector{Matrix}(undef, size(hz_file_paths))

for i in axes(hzs)[1]
    hzs[i] = abs.(readdlm(hz_file_paths[i], ',', Float64, '\n'))
end

index = Observable(1)
fig = Figure()
ax = Axis(fig[1,1], title="Simulated magnetic field", aspect=DataAspect())
heatmap!(fig[1,1], @lift(hzs[$index]), interpolate=true)

if target == :animation
    record(fig, "animation.mp4", axes(hzs)[1]; framerate=10) do i
        index[] = i
    end
else
    for i in axes(hzs)[1]
        index[] = i
        save(hz_file_paths[i] * ".png", fig)
    end
end