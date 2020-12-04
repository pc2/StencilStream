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

    std::deque<std::thread> collection_threads;

    if (!parameters.benchmark_mode)
    {
        std::cout << "Simulating..." << std::endl;

        StencilExecutor<FDTDCell, stencil_radius, n_buffer_columns, n_buffer_rows, FDTD_BURST_SIZE> executor(fpga_queue);
        executor.set_generations(2 * parameters.n_sample_steps);
        executor.set_buffer(FDTDKernel::setup_cell_buffer(fpga_queue));

        for (uindex_t i_frame = 0; i_frame < parameters.n_frames; i_frame++)
        {
            event comp_event = executor.run(FDTDKernel(i_frame * parameters.n_sample_steps, parameters));
            comp_event.wait();

            unsigned long event_start = comp_event.template get_profiling_info<info::event_profiling::command_start>();
            unsigned long event_end = comp_event.template get_profiling_info<info::event_profiling::command_end>();
            std::cout << "Collected frame " << i_frame << ", simulation time: " << (event_end - event_start) / 1000000000.0 << " s" << std::endl;

            collection_threads.push_front(std::thread(SampleCollector(i_frame, executor.get_buffer())));
        }

        std::cout << "Simulation complete!" << std::endl;
        std::cout << "Awaiting frame writeout completion." << std::endl;

        while (!collection_threads.empty())
        {
            collection_threads.back().join();
            collection_threads.pop_back();
        }

        std::cout << "Writeout complete!" << std::endl;
    }
    else
    {
        std::cout << "Starting benchmark" << std::endl;

        double clock_frequency;
        std::cout << "Clock frequency: ";
        std::cin >> clock_frequency;
        std::cout << std::endl;

        std::vector<double> runtimes;

        StencilExecutor<FDTDCell, stencil_radius, n_buffer_columns, n_buffer_rows, FDTD_BURST_SIZE> executor(fpga_queue);
        executor.set_buffer(FDTDKernel::setup_cell_buffer(fpga_queue));
        for (uindex_t i = 0; i < parameters.n_frames; i++)
        {
            executor.set_generations((i + 1) * parameters.n_sample_steps);
            event comp_event = executor.run(FDTDKernel(0, parameters));
            comp_event.wait();

            unsigned long event_start = comp_event.template get_profiling_info<info::event_profiling::command_start>();
            unsigned long event_end = comp_event.template get_profiling_info<info::event_profiling::command_end>();
            runtimes.push_back(double(event_end - event_start) / 1000000000.0);
            std::cout << "Run " << i << " with " << (i + 1) * parameters.n_sample_steps << " passes took " << runtimes.back() << " seconds" << std::endl;
        }

        double delta_passes = parameters.n_sample_steps / pipeline_length;

        // Actually the mean delta seconds per pass.
        double delta_seconds_per_pass = 0.0;
        for (uindex_t i = 0; i < runtimes.size() - 1; i++)
        {
            delta_seconds_per_pass += abs(double(runtimes[i + 1]) - double(runtimes[i])) / delta_passes;
        }
        delta_seconds_per_pass /= runtimes.size() - 1;

        std::cout << "Time per buffer pass: " << delta_seconds_per_pass << "s" << std::endl;

        double loops_per_pass = n_buffer_rows * n_buffer_columns;
        double seconds_per_loop = delta_seconds_per_pass * loops_per_pass;
        double cycles_per_loop = seconds_per_loop * clock_frequency;

        std::cout << "Cycles per Loop, aka II.: " << cycles_per_loop << std::endl;

        double fo_per_core = 118;
        double fo_per_loop = fo_per_core * pipeline_length;
        double fo_per_pass = fo_per_loop * loops_per_pass;
        double fo_per_second = fo_per_pass / delta_seconds_per_pass;

        std::cout << "Raw Performance: " << fo_per_second / 1000000000.0 << " GFLOPS" << std::endl;

        double generations_per_pass = double(pipeline_length) / 2.0;
        double generations_per_second = generations_per_pass / delta_seconds_per_pass;

        std::cout << "Performance: " << generations_per_second << " Generations/s" << std::endl;
    }

    return 0;
}
