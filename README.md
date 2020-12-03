# StencilSteam

Generic Stencil Simulation Library for FPGAs.

This project uses Intel's OneAPI to provide a C++ template library that can execute arbitrary Moore-like stencil kernels on arbitrary cell types in arbitrary buffers, using FPGAs.

## Performance Metrics

Below are performance metrics of some sample applications, as of release [v1.0.0](https://github.com/pc2/StencilStream/releases/tag/v1.0.0). The `conway` application is optimized for readability, not for performance, and is therefore not listed.

### Nallatech/Bittware 520N Board (Stratix 10 GX 2800)

| Application | Main Loop II | Pipeline Depth | Cycle Frequency | Generations per Second | Overall Performance | Logic Usage | Register Usage | RAM Usage | DSP Usage |
|-------------|--------------|----------------|-----------------|------------------------|---------------------|-------------|----------------|-----------|-----------|
| `hotspot`   | 1.05 cycles  | 225 cores      | 79.63 MHz       | 16,328 G/s             | 256.84 GFLOPS       | 85.34%      | 51.23%         | 38.31%    | 58.64%    |
| `fdtd`      | 1.73 cycles  | 30 cores       | 225 MHz         | 233.10 G/s             | 29.02 KFLOPS        | 83.19%      | 50.37%         | 43.91%    | 45.42%    |

### Intel PAC (Stratix 10 SX)

| Application | Main Loop II | Pipeline Depth | Cycle Frequency | Generations per Second | Overall Performance | Logic Usage | Register Usage | RAM Usage | DSP Usage |
|-------------|--------------|----------------|-----------------|------------------------|---------------------|-------------|----------------|-----------|-----------|
| `hotspot`   | 1.06 cycles  | 100 cores      | 225.00 MHz      | 20,161.29 G/s          | 317.17 GFLOPS       | 64.26%      | 35.75%         | 25.09%    | 26.11%    |
| `fdtd`      | 1.45 cycles  | 20 cores       | 218.00 MHz      | 178.95 G/s             | 24.43 KFLOPS        | 69.41%      | 37.87%         | 34.66%    | 30.29%    |

## How to use StencilStream

### Required Software

This library requires the "Intel® oneAPI Base Toolkit for Linux" as well as the "Intel® FPGA Add-On for oneAPI Base Toolkit", which you can download [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html#operatingsystem=Linux&#distributions=Web%20and%20Local%20Install&#options=Online). You also need to have a GCC toolchain with support for C++17 features installed and loaded. If your FPGA accelerator card isn't an Intel® PAC, you also need the board support package of your card.

If you're working with [the Noctua super-computer by the Paderborn Center for Parallel Computing](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua/) and the Nallatech/Bittware 520N Board, you can easily load all required components by executing the following commands:

``` bash
source /cm/shared/opt/intel_oneapi/{latest-version}/setvars.sh
module load nalla_pcie compiler/GCC
```

### A basic stencil kernelcs

As an example, we are going to implement a simple version of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life). However, please note that this isn't the most efficient way to do it, just an example.

First, create a new working directory for your project and copy the `stencil` folder into it. We will only need a single source file, so create a `main.cpp` too! We are now going to walk through it:

``` C++
#include "stencil/stencil.hpp"
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <fstream>
```

These are some required includes. First, we include StencilStream and some SYCL extensions by Intel. Normally, you would now include the namespaces `cl::sycl` and `stencil`, but here, we don't do that to show which types come from which library.

``` C++
using cell = bool;
const uindex_t radius = 1;
const uindex_t width = 1024;
const uindex_t height = 1024;
```

Next are some important definitions: The cell type, the radius of the stencil buffer, and the width and height of the cell buffer. The radius defines how many neighbours of a cell we need to calculate the next generation. In our case, we only need the direct neighbours, so we set the radius to 1. This means that the stencil buffer will be 3 by 3 cells big. Lastly, we need to define the maximum size of the cell buffer or grid. This is important since some optimizations of StencilStream need these values to be hard-coded at compile time.

This is everything we need to define the stencil kernel, so let's do it now:

``` C++
const auto conway = [](stencil::Stencil2D<cell, radius> const &stencil, stencil::Stencil2DInfo const &info) {
    stencil::UID idx = info.center_cell_id;
``` 
As you can see, a stencil kernel simply is an invocable object. In this case, we've chosen a lambda expression, but more complicated applications may define thier stencil kernel as a class with an `operator()` method.

The first argument is the stencil buffer itself and the second argument is a struct with useful information about the current invocation. For now, this struct provides only one useful information: The coordinates of the central cell of the stencil buffer, the cell we are going to replace.
``` C++
    stencil::index_t lower_h_bound, lower_v_bound;
    lower_h_bound = lower_v_bound = -radius;
    stencil::index_t upper_h_bound, upper_v_bound;
    upper_h_bound = upper_v_bound = radius;

    if (idx.c == 0)
    {
        lower_h_bound = 0;
    }
    else if (idx.c == width - 1)
    {
        upper_h_bound = 0;
    }

    if (idx.r == 0)
    {
        lower_v_bound = 0;
    }
    else if (idx.r == height - 1)
    {
        upper_v_bound = 0;
    }
```
Our first action is to identify the neighbours we are actually allowed to look at. This is a little bit more complicated since StencilStream does not make any promises about the border cells, the cells that are outside of the actual cell buffer. The stencil buffer still contains values for them due to technical reasons, but a user must not use them since any update may break their behavior. In our case, we update our bounds and simply ignore cells outside of the cell buffer.
``` C++
    uint8_t alive_neighbours = 0;
#pragma unroll
    for (stencil::index_t c = -radius; c <= stencil::index_t(radius); c++)
    {
#pragma unroll
        for (stencil::index_t r = -radius; r <= stencil::index_t(radius); r++)
        {
            bool is_valid_neighbour = (c != 0 || r != 0) &&
                                      (c >= lower_h_bound) &&
                                      (c <= upper_h_bound) &&
                                      (r >= lower_v_bound) &&
                                      (r <= upper_v_bound);

            if (is_valid_neighbour && stencil[stencil::ID(c, r)])
            {
                alive_neighbours += 1;
            }
        }
    }
```
Next, we count our living neighbours since their numbers decides the fate of our cell. The `for`-loops for that are completely unrolled, which means that these evaluations will be carried out in parallel.
``` C++
    if (stencil[stencil::ID(0, 0)])
    {
        return alive_neighbours == 2 || alive_neighbours == 3;
    }
    else
    {
        return alive_neighbours == 3;
    }
};
```
Now we know how many of our neighbours are alive and can therefore return the new cell value, according to [the rules of the game](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Rules).

The only thing left is to execute the stencil kernel. We do this like this:
``` C++
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <#Generations>" << std::endl;
        return 1;
    }

#ifdef HARDWARE
    cl::sycl::INTEL::fpga_selector device_selector;
#else
    cl::sycl::INTEL::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue fpga_queue(device_selector);

    cl::sycl::buffer<cell, 2> cell_buffer = read();

    stencil::StencilExecutor<cell, radius, width, height> executor(fpga_queue);
    executor.set_buffer(cell_buffer);
    executor.set_generations(std::stoi(argv[1]));
    executor.run(conway);

    write(cell_buffer);

    return 0;
}
```
After checking the arguments, we use the `HARDWARE` flag to either use the emulation device or the FPGA. With this device, we create a job queue. With OpenCL, this is a daunting task to get right, but here it's only six lines of simple code that also handles all the errors that might come up.

Then, we allocate the cell buffer and write the initial state to it. The actual process isn't important for this example and is therefore left out. The same is true when the results are written back .

When all of this is done, we create a stencil executor and configure it with the cell type, the radius of the stencil buffer, the width and height of the cell buffer, the cell buffer itself and the number of generations. When configured, we pass the kernel to the executor, which then submits the kernel for execution and waits until it's done. We can find the results in the original cell buffer.

That's it. This is all of the code you have to write. Everything else, like getting cells to and from the global buffer, caching intermediate values, or resolving loop dependencies is done by StencilStream. You only need to provide the stencil kernel, everything else is handled for you.

One last thing we have to talk about is the build environment. The usual and recommended way to control a StencilStream application is to create a Makefile and set appropriate definitions. You can use the following `Makefile` as an example:

``` Makefile
CC = dpcpp
STENCIL_PATH = ./

ARGS = -fintelfpga -Xsv -Xsfpc -std=c++17 -I$(STENCIL_PATH) 

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifndef PIPELINE_LEN
	PIPELINE_LEN=10
endif
ARGS += -DSTENCIL_PIPELINE_LEN=$(PIPELINE_LEN)

ifdef HARDWARE
	ARGS += -DHARDWARE
	ARGS += -Xshardware
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

conway: clean conway.cpp Makefile
	$(CC) $(ARGS) conway.cpp -o conway

conway.report.tar.gz: clean conway.cpp Makefile
	$(CC) $(ARGS) -fsycl-link -Xshardware conway.cpp -o conway
	tar -caf conway.report.tar.gz conway.prj/reports

clean:
	git clean -dXf
```
If you just do `make conway`, an emulation image will be created that can be executed to verify your code. If you want to synthesize the design for your FPGA, you have to set the environment variable `HARDWARE`. However, this might take a lot of time and therefore, you should first generate a design report to evaluate th performance of the design and estimate the compilation time. You do this by running `make conway.report.tar.gz`.

Another useful and important environment variable is `PIPELINE_LEN`. It controls how often your stencil kernel will be replicated for parallel execution. A higher `PIPELINE_LEN`, leads to higher parallelity and therefore overall speed, but also to higher resource use and therefore longer compilation times and lower clocking frequencies. Tweak this value when you're don with your design and want to get the ultimate performance.

### Going further

This clearly isn't everything. Especially with the example of Conway's Game of Life, you might want to process multiple cell by one stencil kernel invocation to enhance parallelity even further, or you might want to export intermediate values. In this case, you can take a look at the [FDTD example](examples/fdtd/), which applies some techniques for that.
