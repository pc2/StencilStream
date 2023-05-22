using DataFrames
using CSV
using Statistics

iteration_re = r"it = ([0-9]+) \(iter = ([0-9]+), time = ([^)]+)\)"
total_re = r"Total time = (.+)$"

log_entries = DataFrame(i_iteration=Int[], pseudo_steps=Int[], runtime=Float64[], throughput=Float64[])
total_runtime = nothing

while isopen(stdin)
    line = readline(stdin)
    if length(line) == 0
        continue
    end
    
    if (line_match = match(iteration_re, line)) !== nothing
        i_iteration = parse(Int, line_match[1])
        pseudo_steps = parse(Int, line_match[2])
        runtime = parse(Float64, line_match[3])
        throughput = pseudo_steps / runtime

        push!(log_entries, (i_iteration, pseudo_steps, runtime, throughput))
    elseif (line_match = match(total_re, line)) !== nothing
        global total_runtime = parse(Float64, line_match[1])

    else
        println(stderr, "Unable to parse line \"$line\", ignoring...")
        continue
    end
end

target_name = ARGS[1]

CSV.write("performance_$target_name.csv", log_entries)

println("runtime_seconds{example=\"convection\",target=\"$target_name\"} $total_runtime")

mean_throughput = mean(log_entries[:, :throughput])
std_throughput = std(log_entries[:, :throughput])
min_throughput = minimum(log_entries[:, :throughput])
max_throughput = maximum(log_entries[:, :throughput])
println("throughput_mean_steps_per_second{example=\"convection\",target=\"$target_name\"} $mean_throughput")
println("throughput_std_steps_per_second{example=\"convection\",target=\"$target_name\"} $std_throughput")
println("throughput_min_steps_per_second{example=\"convection\",target=\"$target_name\"} $min_throughput")
println("throughput_max_steps_per_second{example=\"convection\",target=\"$target_name\"} $max_throughput")

overhead = total_runtime - sum(log_entries[:, :runtime])
println("overhead_seconds{example=\"convection\",target=\"$target_name\"} $overhead")
println("overhead_seconds_per_step{example=\"convection\",target=\"$target_name\"} $(overhead/ size(log_entries,1))")
