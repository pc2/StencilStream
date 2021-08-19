# StencilStream

Generic Stencil Simulation Library for FPGAs.

This project uses Intel's OneAPI to provide a C++ template library that can execute arbitrary Moore-like stencil kernels on arbitrary cell types in arbitrary buffers, using FPGAs.

## Performance Metrics

To be done.

## How to use StencilStream

### Required Software

This library requires the "Intel® oneAPI Base Toolkit for Linux" as well as the "Intel® FPGA Add-On for oneAPI Base Toolkit", which you can download [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html#operatingsystem=Linux&#distributions=Web%20and%20Local%20Install&#options=Online). You also need to have a GCC toolchain with support for C++17 features installed and loaded as well the [boost libraries](https://www.boost.org), version 1.29.0 or newer. If your FPGA accelerator card isn't an Intel® PAC, you also need the board support package of your card.

If you're working with [the Noctua super-computer by the Paderborn Center for Parallel Computing](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua/) and the Nallatech/Bittware 520N Board, you can easily load all required components by executing the following commands:

``` bash
source /cm/shared/opt/intel_oneapi/{latest-version}/setvars.sh
module load nalla_pcie compiler/GCC
```

### A basic stencil kernel

As an example, we are going to implement a simple version of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life). However, please note that this isn't the most efficient way to do it, just an example.

First, create a new working directory for your project and copy the `StencilStream` folder into it. We will only need a single source file, so create a `conway.cpp` too! We are now going to walk through it:

``` C++
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/StencilExecutor.hpp>
```

First, we include StencilStream and some SYCL extensions by Intel. Normally, you would use the namespaces `cl::sycl` and `stencil`, but here we don't do that to show you which types come from which library.

``` C++
using Cell = bool;
const Cell halo_value = false;
const stencil::uindex_t stencil_radius = 1;
```

Next are some important definitions: The cell type, the value of cells in the grid halo, and the radius of the stencil buffer. In our example, a cell is either alive or dead. We express that with a boolean value which is true if the cell is alive and false if it is dead. The cells are arranged in a grid, but in order to update the cells on the borders of the grid, we need cells *outside* of the grid. StencilStream assures that these cells always have a constant halo value. If this halo value and the transition function is well-chosen, we don't have to do any edge handling. Here, we assume that cells outside of the grid to be always dead, so we pick the halo value `false`. The radius of the stencil defines how many neighbors of a cell we need to calculate the next generation. In our case, we only need the direct neighbors, so we set the radius to 1. This means that the stencil buffer will be 3 by 3 cells big.

This is everything we need to define the transition function, so let's do it now:

``` C++
auto conway = [](stencil::Stencil<Cell, stencil_radius> const &stencil) {
``` 

As you can see, a transition function is just an invocable object. In this case, we have chosen a lambda expression, but more complicated applications may define their transition function as a class with an `operator()` method.

The only argument is the stencil buffer itself. It also contains useful information about the current invocation, like the coordinates of the central cell of the stencil buffer. This is the cell we are going to replace.

``` C++
    stencil::ID idx = stencil.id;

    uint8_t alive_neighbours = 0;
#pragma unroll
    for (stencil::index_t c = -stencil_radius; c <= stencil::index_t(stencil_radius); c++) {
#pragma unroll
        for (stencil::index_t r = -stencil_radius; r <= stencil::index_t(stencil_radius); r++) {
            if (stencil[stencil::ID(c, r)] && !(c == 0 && r == 0)) {
                alive_neighbours += 1;
            }
        }
    }
```

First, we count the living neighbors since their numbers decides the fate of our cell. The `for`-loops for that are completely unrolled, which means that these evaluations will be carried out in parallel.

``` C++
    if (stencil[stencil::ID(0, 0)]) {
        return alive_neighbours == 2 || alive_neighbours == 3;
    } else {
        return alive_neighbours == 3;
    }
};
```

Now we know how many of our neighbors are alive and can therefore return the new cell value according to [the rules of the game](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Rules).

``` C++
cl::sycl::buffer<Cell, 2> read(stencil::uindex_t width, stencil::uindex_t height) {
    cl::sycl::buffer<Cell, 2> input_buffer(cl::sycl::range<2>(width, height));
    auto buffer_ac = input_buffer.get_access<cl::sycl::access::mode::write>();

    for (stencil::uindex_t r = 0; r < height; r++) {
        for (stencil::uindex_t c = 0; c < width; c++) {
            char Cell;
            std::cin >> Cell;
            assert(Cell == 'X' || Cell == '.');
            buffer_ac[c][r] = Cell == 'X';
        }
    }

    return input_buffer;
}

void write(cl::sycl::buffer<Cell, 2> output_buffer) {
    auto buffer_ac = output_buffer.get_access<cl::sycl::access::mode::read>();

    stencil::uindex_t width = output_buffer.get_range()[0];
    stencil::uindex_t height = output_buffer.get_range()[1];

    for (stencil::uindex_t r = 0; r < height; r++) {
        for (stencil::uindex_t c = 0; c < width; c++) {
            if (buffer_ac[c][r]) {
                std::cout << "X";
            } else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
}
```

The next part is some boilerplate code to read the input from stdin and write the output to stdout. Nothing to spectacular.

The only thing left is to run the calculations. We do this like this:

``` C++
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <n_generations>" << std::endl;
        return 1;
    }

    stencil::uindex_t width = std::stoi(argv[1]);
    stencil::uindex_t height = std::stoi(argv[2]);
    stencil::uindex_t n_generations = std::stoi(argv[3]);

    cl::sycl::buffer<Cell, 2> grid_buffer = read(width, height);

    using Executor = stencil::StencilExecutor<Cell, stencil_radius, decltype(conway)>;
    Executor executor(halo_value, conway);
    executor.set_input(grid_buffer);
```

After checking and parsing the arguments, we read the input data and initialize the executor. This is the central API facade to control the calculations. In it's simplest form, it only requires cell type, the radius of the stencil and the type of the transition function as template arguments. It has more template arguments, but these are performance parameters. We are looking into them later. The actual constructor arguments are only the initial data, the halo value and an instance of the transition function.

``` C++
#ifdef HARDWARE
    executor.select_fpga();
#else
    executor.select_emulator();
#endif
```

Next, we use the `HARDWARE` flag to either pick the emulator or the FPGA. With OpenCL, this is a daunting task to get right, but here there are only some lines of simple code that also handle all of the errors that might come up.

``` C++
    executor.run(n_generations);

    executor.copy_output(grid_buffer);
    write(grid_buffer);

    return 0;
}
```

When all of this is done, we just tell the executor to calculate the requested number of generations. After that, we copy the results back to the grid buffer and write them to stdout.

That's it. This is all of the code you have to write. Everything else, like getting cells to and from the global buffer, caching intermediate values, or resolving loop dependencies is done by StencilStream. You only need to provide the transition function and some info about it, everything else is handled for you.

One last thing we have to talk about is the build environment. The usual and recommended way to build a StencilStream application is to create a Makefile and set appropriate definitions. You can use the following `Makefile` as an example:

``` Makefile
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -fintelfpga -Xsv -std=c++17 -I$(STENCIL_STREAM_PATH) -O3

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

EMU_ARGS = $(ARGS)
HW_ARGS = $(ARGS) -DHARDWARE -Xshardware 

conway_emu: conway.cpp Makefile
	$(CC) $(EMU_ARGS) conway.cpp -o conway_emu

conway_hw: conway.cpp Makefile
	$(CC) $(HW_ARGS) conway.cpp -o conway_hw

conway_hw.report.tar.gz: conway.cpp Makefile
	rm -f conway_hw
	$(CC) $(HW_ARGS) -fsycl-link conway.cpp -o conway_hw
	tar -caf conway_hw.report.tar.gz conway_hw.prj/reports

clean:
	git clean -dXf
```
If you just run `make conway_emu`, an emulation image will be created that can be executed on the CPU to verify your code. If you want to synthesize the design for your FPGA, you have to run `make conway_hw`. However, this might take a lot of time and therefore, you should first generate a design report to evaluate the performance of the design and estimate the compilation time. You do this by running `make conway_hw.report.tar.gz`.

### Going further

This example only showed the general way StencilStream is used. More optimised and non-trivial examples can be found in the [examples folder](https://github.com/pc2/StencilStream/tree/master/examples). However, in order to fully understand the way StencilStream works and to optimize your application, you should take a look at [the documentation](https://pc2.github.io/StencilStream/index.html) and especially at the [architecture document](https://pc2.github.io/StencilStream/Architecture.html). Future releases will also feature an optimization guide that discusses the different optimization parameters and gives advice for good designs.