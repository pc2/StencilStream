#!/usr/bin/env -S julia --project
using DelimitedFiles
using Statistics
using Glob
using ColorSchemes
using FileIO
using Base.Threads

if size(ARGS,1) < 1
    println(stderr, "Usage: " * PROGRAM_FILE * " <output directory>")
    exit(1)
end

output_directory_path = ARGS[1]
if !isdir(output_directory_path)
    println("Output directory "* output_directory_path * " does not exist or is not a directory!")
    exit(1)
end

@threads for path in glob("*.csv", output_directory_path)
    data = readdlm(path, ',', Float64)
    max_abs_value = maximum(abs.(data))
    if max_abs_value == 0
        norm_data = data .+ 0.5
    else
        norm_data = data ./ (2 * max_abs_value) .+ 0.5
    end
    image = (cell -> get(ColorSchemes.broc, cell)).(norm_data)
    save(path * ".png", image)
end