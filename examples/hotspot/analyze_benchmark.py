#!/usr/bin/env python3
"""
# How to use this script.

First, run the benchmark.sh and verify_benchmark.sh scripts and pipe their output into `benchmark.py`
and `verify_benchmark.py` respectively. If you submit the scripts as slurm jobs, this is done
automatically. The output is formatted like a Python dict, but you need to remove error and output
messages. Lastly, you need to copy the cycle frequency and the pipeline length from the reports into this script. 
"""

from benchmark import runtime

cycle_frequency = 79.63 * 10**6
pipeline_length = 225
step_size = 900

height = width = 1024
radius = 1
flo_per_cell = 15
cell_size = 2 * 4

# Analysis
for i in runtime.keys():
    print("PIPELINE_LEN={}:".format(i))
    
    original_runtime = runtime[i]
    offset_runtime = [original_runtime[0]] + original_runtime
    
    delta = [current_sample - next_sample for (current_sample, next_sample) in zip(original_runtime, offset_runtime)]
    mean_delta = (sum(delta) / len(delta)) * (pipeline_length / step_size)
    
    print("\tDelta Time per Grid Pass: {:.2f} milliseconds".format(mean_delta * 10**3))
    
    overhead = runtime[i][0] - mean_delta
    print("\tOverhead: {:.2f} seconds".format(overhead))
    
    executed_flop = pipeline_length * flo_per_cell * height * width
    executed_cycles = mean_delta * cycle_frequency
    
    gflops = executed_flop / mean_delta * 10**-9
    ii = executed_cycles / (width * height)
    
    print("\tOverall performance: {:.2f} GFLOPS".format(gflops))
    print("\tApparent II / Cycles per Grid Cell: {:.2f} cycles".format(ii))
    
    throughput = (2 * cell_size * width * height) / mean_delta * 10**-9
    
    print("\tGlobal Memory Throughput: {:.2f} GB/s".format(throughput))