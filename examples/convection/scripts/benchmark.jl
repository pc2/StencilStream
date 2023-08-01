#!/usr/bin/env -S julia --project=../..
using DataFrames
using CSV
using Statistics
using JSON

const N_SUBGENERATIONS = 3
const N_REPLICATIONS = 5
const N_CUS = N_SUBGENERATIONS * N_REPLICATIONS
const OPERATIONS_PER_CELL = (5+5+3+6+6+6) + (10+3+2+14+3+2) + 2
const CELL_SIZE = 128 # bytes, with padding

function parse_report_js(path)
    fields = Dict{String,Any}()
    for line in readlines(path)
        if occursin("fileJSON", line)
            continue # Avoiding to load all source files. They're too big for REs.
        end

        matched_line = match(r"var (.+)=(\{.+\});$", line)
        if matched_line === nothing
            continue
        end
        
        name = matched_line[1]
        data = replace(matched_line[2], "\\'" => "'")
        fields[name] = JSON.parse(data)
    end
    return fields
end

function load_report_details(report_path)
    quartus_data = parse_report_js("$report_path/resources/quartus_data.js")
    f = parse(Float32, quartus_data["quartusJSON"]["quartusFitClockSummary"]["nodes"][1]["kernel clock fmax"]) * 1e6

    report_data = parse_report_js("$report_path/resources/report_data.js")
    kernels = report_data["loop_attrJSON"]["nodes"]
    loops = Iterators.flatmap(kernel -> kernel["children"], kernels)
    loop_latency = sum(Iterators.map(loop -> parse(Float32, loop["lt"]), loops))

    return f, loop_latency
end

function model_throughput(f, loop_latency, n_rows, n_cols)
    cu_latency = n_rows + 1
    pipeline_latency = N_CUS * cu_latency
    n_iterations = pipeline_latency + (n_rows * n_cols)
    stall_ratio = 0.5
    n_cycles = n_iterations / stall_ratio  + loop_latency
    work = (n_rows * n_cols) * N_REPLICATIONS
    work / n_cycles * f
end

function analyze_log(logfile)
    iteration_re = r"it = ([0-9]+) \(iter = ([0-9]+), time = ([^)]+)\)"
    total_re = r"Total time = (.+)$"

    e2e_runtime = nothing
    pseudo_transient_runtimes = DataFrame(i_iteration=Int[], pseudo_steps=Int[], runtime=Float64[])

    while e2e_runtime === nothing
        line = readline(logfile)
        if length(line) == 0
            continue
        end
        
        if (line_match = match(iteration_re, line)) !== nothing
            i_iteration = parse(Int, line_match[1])
            pseudo_steps = parse(Int, line_match[2])
            runtime = parse(Float64, line_match[3])

            push!(pseudo_transient_runtimes, (i_iteration, pseudo_steps, runtime))
        elseif (line_match = match(total_re, line)) !== nothing
            e2e_runtime = parse(Float64, line_match[1])

        else
            println(stderr, "Unable to parse line \"$line\", ignoring...")
            continue
        end
    end

    return e2e_runtime, pseudo_transient_runtimes
end

function resolution_exploration()
    exec = ARGS[2]

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
    
        experiment = copy(experiment_template)
        experiment["res"] = res
        open(experiment_path, "w") do experiment_file 
            JSON.print(experiment_file, experiment)
        end
    
        command = `$exec $experiment_path out`
    
        open(command, "r") do process_in
            e2e_runtime, invocation_runtimes = analyze_log(process_in)
            push!(e2e_runtimes, (res, e2e_runtime))
            invocation_runtimes.res = [res for _ in eachrow(invocation_runtimes)]
            append!(pseudo_transient_runtimes, invocation_runtimes)
        end
    
        CSV.write("pseudo_transient_perf.csv", pseudo_transient_runtimes)
        CSV.write("e2e_perf.csv", e2e_runtimes)
    end
    
    Base.Filesystem.rm(experiment_path)
    nothing
end

function default_benchmark()
    exe = ARGS[2]
    report_path = exe * ".prj/reports"
    #f, loop_latency = load_report_details(report_path)
    f = 350e6
    loop_latency = 2000

    experiment_path = "experiments/default.json"
    experiment_data = JSON.parsefile(experiment_path)
    ly = experiment_data["ly"]
    lx = experiment_data["lx"]
    res = experiment_data["res"]

    out_path = Base.Filesystem.mkpath("./out/")
    command = `$exe $experiment_path $out_path`

    open(command, "r") do process_in
        _, pseudo_transient_runtimes = analyze_log(process_in)
        CSV.write("pseudo_transient_runtimes.csv", pseudo_transient_runtimes)

        measured_performance = maximum((ly * lx * res^2) .* pseudo_transient_runtimes.pseudo_steps ./ pseudo_transient_runtimes.runtime)
        model_performance = model_throughput(f, loop_latency, ly * res, lx * res)
        max_peak_performance = f * N_REPLICATIONS
        occupancy = measured_performance / max_peak_performance
        model_accurracy = measured_performance / model_performance

        gflops = OPERATIONS_PER_CELL * measured_performance * 1e-9
        mem_throughput = CELL_SIZE * measured_performance / N_CUS

        metrics = Dict(
            "n_cus" => N_CUS,
            "f" => f,
            "occupancy" => occupancy,
            "measured" => measured_performance,
            "accuracy" => model_accurracy,
            "GFLOPS" => gflops,
            "mem_throughput" => mem_throughput
        )
        open("metrics.json", "w") do metrics_file
            JSON.print(metrics_file, metrics)
        end
    end
end

mode = ARGS[1]
if mode == "res-exploration"
    resolution_exploration()
else mode == "default"
    default_benchmark()
end