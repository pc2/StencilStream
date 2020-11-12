# Implementation of the Rodinia Hotspot benchmark

Since many parameters are fixed, the compiled program accepts less arguments than the original benchmark program:
* `<sim_time>`: Number of iterations
* `<temp_file>`: Name of the file containing the initial temperature values of each cell. One is provided in [data/temp_1024](data/temp_1024).
* `<power_file>`: Name of the file containing the dissipated power values of each cell. One is provided in [data/power_1024](data/power_1024).
* `<output_file>`: Name of the output file.

Builds of the example are uploaded as [Git tags](https://git.uni-paderborn.de/joo/sycl-stencil/-/tags) to the Uni's Gitlab instance. These tarballs only contain the program and the synthesis report. Full output folders are found on Noctua in a [custom folder](/upb/departments/pc2/groups/hpc-prf-hpmesh/joo/sycl-stencil).