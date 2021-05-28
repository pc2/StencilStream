/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "collection.hpp"
#include "simulation.hpp"
#include <StencilStream/StencilExecutor.hpp>
#include <deque>

auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (cl::sycl::exception const &e)
        {
            std::cout << "Caught asynchronous SYCL exception:\n"
                      << e.what() << "\n";
            std::terminate();
        }
    }
};

int main(int argc, char **argv)
{
    Parameters parameters(argc, argv);

#ifdef HARDWARE
    INTEL::fpga_selector device_selector;
#else
    INTEL::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue fpga_queue(device_selector, exception_handler, {property::queue::enable_profiling{}});

    std::deque<std::thread> collection_threads;

    std::cout << "Simulating..." << std::endl;

    uindex_t grid_width = 2 * (parameters.disk_radius + 1);
    uindex_t grid_height = grid_width / vector_len;
    if (grid_width % vector_len != 0)
    {
        grid_height += 1;
    }

    std::cout << "Simulation grid size: " << grid_width << "x" << grid_height * vector_len << " cells" << std::endl;

    StencilExecutor<FDTDCell, stencil_radius, FDTDKernel, pipeline_length, tile_width, tile_height, FDTD_BURST_SIZE> executor(grid_width, grid_height, FDTDKernel::halo(), FDTDKernel(parameters));
    executor.set_queue(fpga_queue, true);

    for (uindex_t i_frame = 0; i_frame < parameters.n_frames; i_frame++)
    {
        executor.run(2 * parameters.n_sample_steps);

        buffer<FDTDCell, 2> out_buffer(range<2>(grid_width, grid_height));
        executor.copy_output(out_buffer);

        std::cout << "Collected frame " << i_frame << ", simulation time: " << executor.get_runtime_sample()->get_total_runtime() << " s" << std::endl;

        collection_threads.push_front(std::thread(SampleCollector(i_frame, out_buffer)));
    }

    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Awaiting frame writeout completion." << std::endl;

    while (!collection_threads.empty())
    {
        collection_threads.back().join();
        collection_threads.pop_back();
    }

    std::cout << "Writeout complete!" << std::endl;

    return 0;
}
