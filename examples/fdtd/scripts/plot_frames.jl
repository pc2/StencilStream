#!/usr/bin/env -S julia --project
# Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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