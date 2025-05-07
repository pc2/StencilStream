#!/usr/bin/env -S julia --project
using CairoMakie

data = open(io -> collect(readeach(io, Float32)), ARGS[1])
side_length = Int(√size(data)[1])
data = reshape(data, (side_length, side_length))
fig = heatmap(data)
save(ARGS[2], fig)
