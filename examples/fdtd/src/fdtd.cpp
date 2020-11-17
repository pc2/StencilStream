/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "collection.hpp"
#include "simulation.hpp"

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

    std::deque<std::thread> collection_threads = SampleCollector::start_collecting(fpga_queue, parameters);

    std::cout << "Simulating..." << std::endl;

    StencilExecutor<FDTDCell, stencil_radius, n_buffer_columns, n_buffer_rows, FDTD_BURST_SIZE> executor(fpga_queue);
    executor.set_buffer(FDTDKernel::setup_cell_buffer(fpga_queue));
    executor.set_generations(2 * parameters.n_time_steps);
    event comp_event = executor.run(FDTDKernel(parameters));

    std::cout << "Simulation complete!" << std::endl;

    unsigned long event_start = comp_event.get_profiling_info<info::event_profiling::command_start>();
    unsigned long event_end = comp_event.get_profiling_info<info::event_profiling::command_end>();
    std::cout << "Total simulation time: " << (event_end - event_start) / 1000000000.0 << " s" << std::endl;

    std::cout << "Awaiting frame writeout completion." << std::endl;

    while (!collection_threads.empty())
    {
        collection_threads.back().join();
        collection_threads.pop_back();
    }

    std::cout << "Writeout complete!" << std::endl;

    return 0;
}
