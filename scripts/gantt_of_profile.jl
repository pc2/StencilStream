#!/usr/bin/env -S julia --project
# A simple script that loads the contents of a AOCL profile.json file and renders the kernel events in a GANTT chart.
# It takes the path to the profile.json file as its first argument and the path for the resulting plot as the second argument.
using CairoMakie
using JSON

function gantt_of_profile(in_path, out_path)
    data = open(JSON.parse, in_path)
    fig = Figure(size=(6400, 400))
    ax = Axis(fig[1,1], ylabel="Queue", xlabel="Time [ms]")

    kernel_events = data["kernels"]["nodes"]
    memtransfer_events = data["memtransfers"]["nodes"]

    queues(events) = [parse(Int, event["command_queue_id"][1]) for event in events]
    start_times(events) = [parse(Int, event["start_time"]) ./ 1e6 for event in events]
    end_times(events) = [parse(Int, event["end_time"]) ./ 1e6 for event in events]

    time_offset = min(minimum(start_times(kernel_events)), minimum(start_times(memtransfer_events)))

    plot_events!(ax, events) = barplot!(ax, queues(events), end_times(events) .- time_offset, fillto=start_times(events) .- time_offset, strokecolor=:black, strokewidth=1, direction=:x)

    plot_events!(ax, kernel_events)
    plot_events!(ax, memtransfer_events)

    save(out_path, fig)
end

if length(ARGS) != 2
    println("Usage: $(PROGRAM_FILE) <input/profile.json> <output/profile.png>")
    println("See top comment of script for more info.")
    exit(1)
else
    gantt_of_profile(ARGS[1], ARGS[2])
end