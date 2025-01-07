#!/usr/bin/env -S julia --project
# A simple script that loads the contents of a AOCL profile.json file and renders the kernel events in a GANTT chart.
# It takes the path to the profile.json file as its first and only argument. The resulting PNG will have the same name
# and path, except for the file
using CairoMakie
using JSON

function gantt_of_profile(in_path, out_path)
    data = open(JSON.parse, in_path)
    fig = Figure()
    ax = Axis(fig[1,1], ylabel="Queue", xlabel="Time [ms]")

    kernel_events = data["kernels"]["nodes"]
    event_queues = (event -> parse(Int, event["command_queue_id"][1])).(kernel_events)
    event_start_times = (event -> parse(Int, event["start_time"])).(kernel_events) ./ 1e6
    event_end_times = (event -> parse(Int, event["end_time"])).(kernel_events) ./ 1e6

    time_offset = minimum(event_start_times)
    event_start_times .-= time_offset
    event_end_times .-= time_offset

    barplot!(ax, event_queues, event_end_times, fillto=event_start_times, strokecolor=:black, strokewidth=1, direction=:x)

    save(out_path, fig)
end

if length(ARGS) != 2
    println("Usage: $(PROGRAM_FILE) <input/profile.json> <output/profile.png>")
    println("See top comment of script for more info.")
    exit(1)
else
    gantt_of_profile(ARGS[1], ARGS[2])
end