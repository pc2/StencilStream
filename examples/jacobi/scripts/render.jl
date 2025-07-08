#!/usr/bin/env -S julia --project
using ColorSchemes
using FileIO
using Statistics

data = open(io -> collect(readeach(io, Float32)), ARGS[1])
side_length = Int(√size(data)[1])
data = reshape(data, (side_length, side_length))
image = (cell -> get(ColorSchemes.inferno, cell)).(data)
save(ARGS[2], image)
