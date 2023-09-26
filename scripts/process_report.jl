#!/usr/bin/env -S julia --project
include("benchmark-common.jl")

if size(ARGS, 1) != 1
    usage = """
    Usage: $PROGRAM_FILE <path to project directory>

    A script to parse and export an Intel OneAPI hardware synthesis report.
    The report viewing page does not necessarily show all kernels if their names in the report data is the same.
    This is however often the case when template classes are used. For example, templated kernel classes get their 
    names shortened to just ">" and lambda kernels inside template classes can not be named since they would otherwise collide.
    However, all of the data is still present in the actual dataset, it only needs to be displayed properly. This script
    loads the data from the report dataset and creates CSV tables that mimic the pages of the report viewer. It's not a
    perfect solution, I wouldn't even call it good, but it's the only way to actually get access to the data without a
    major development effort.
    """
    print(stderr, usage)
    exit(1) 
end

# Load data

raw_data = parse_report_js(ARGS[1] * "/reports/resources/report_data.js")
open("raw_data.json", "w") do f
    JSON.print(f, raw_data)
end

# Generate the loop report

loop_report = DataFrame(loop=String[], code_position=String[], pipelined=String[], II=String[])

function push_loop!(loop_report, loop, level=0)
    loop_name = loop["name"]
    pipelined = loop["data"][1]
    filename = basename(loop["debug"][1][1]["filename"])
    line = loop["debug"][1][1]["line"]
    ii = loop["data"][2]
    push!(loop_report, ["-> "^level * loop_name "$filename:$line" pipelined ii])
    for subloop in loop["children"]
        push_loop!(loop_report, subloop, level+1)
    end
end

for kernel in raw_data["loopsJSON"]["children"]
    for loop in kernel["children"]
        push_loop!(loop_report, loop)
    end
end

CSV.write("loops.csv", loop_report)

# Generate the hardware usage report 

area_overview = DataFrame(unit=String[], type=String[], alut=Real[], ff=Real[], ram=Real[], mlab=Real[], dsp=Real[])
push!(area_overview, vcat(["Max. resources", "info"], raw_data["areaJSON"]["max_resources"]))
push!(area_overview, vcat(["Total usage", "info"], raw_data["areaJSON"]["total"]))
function push_area!(area_report, node; cutoff=Inf, level=0)
    if level >= cutoff
        return
    end

    resources = nothing
    if node["type"] == "function"
        resources = get(node, "total_kernel_resources", nothing)
    elseif node["type"] == "resource"
        resources = get(node, "data", nothing)
    elseif node["type"] == "partition"
        resources = get(node["children"][1], "data", nothing)
    end

    if resources !== nothing
        push!(area_report, vcat(["-> "^level * node["name"], node["type"]], resources))
    end

    for child in get(node, "children", [])
        push_area!(area_report, child; cutoff=cutoff, level=level+1)
    end
end
for (i, toplevel_node) in enumerate(raw_data["areaJSON"]["children"])
    push_area!(area_overview, toplevel_node, cutoff=1, level=0)
    if toplevel_node["type"] == "function"
        function_area = DataFrame(unit=String[], type=String[], alut=Real[], ff=Real[], ram=Real[], mlab=Real[], dsp=Real[])
        push_area!(function_area, toplevel_node)
        CSV.write("area_$(toplevel_node["name"])_$(i).csv", function_area)
    end
end
CSV.write("area_overview.csv", area_overview)