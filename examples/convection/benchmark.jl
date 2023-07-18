using DataFrames
using CSV
using Statistics
using JSON

exec = ARGS[1]

iteration_re = r"it = ([0-9]+) \(iter = ([0-9]+), time = ([^)]+)\)"
total_re = r"Total time = (.+)$"

experiment_template = Dict(
    "ly" => 1.0,
    "lx" => 1.0,
    "py" => 0.5,
    "px" => 0.5,
    "res" => 32,

    "eta0" => 1.0,
    "DcT" => 1.0,
    "deltaT" => 1.0,
    "Ra" => 1e7,
    "Pra" => 1e3,

    "iterMax" => 50000,
    "nt" => 1000,
    "nout" => 10,
    "nerr" => 100,
    "epsilon" => 1e-4,
    "dmp" => 2
)
experiment_path, _ = Base.Filesystem.mktemp(cleanup=false)

pseudo_transient_runtimes = DataFrame(res=Int[], i_iteration=Int[], pseudo_steps=Int[], runtime=Float64[])
e2e_runtimes = DataFrame(res=Int[], runtime=Float64[])

for res in 32:32:512
    println("Running with res $res...")
    e2e_runtime = nothing

    experiment = copy(experiment_template)
    experiment["res"] = res
    open(experiment_path, "w") do experiment_file 
        JSON.print(experiment_file, experiment)
    end

    command = `$exec $experiment_path out`

    open(command, "r") do process_in
        while e2e_runtime === nothing
            line = readline(process_in)
            if length(line) == 0
                continue
            end
            
            if (line_match = match(iteration_re, line)) !== nothing
                i_iteration = parse(Int, line_match[1])
                pseudo_steps = parse(Int, line_match[2])
                runtime = parse(Float64, line_match[3])

                push!(pseudo_transient_runtimes, (res, i_iteration, pseudo_steps, runtime))
            elseif (line_match = match(total_re, line)) !== nothing
                e2e_runtime = parse(Float64, line_match[1])

            else
                println(stderr, "Unable to parse line \"$line\", ignoring...")
                continue
            end
        end
    end

    push!(e2e_runtimes, (res, e2e_runtime))
    CSV.write("pseudo_transient_perf.csv", pseudo_transient_runtimes)
    CSV.write("e2e_perf.csv", e2e_runtimes)
end

Base.Filesystem.rm(experiment_path)