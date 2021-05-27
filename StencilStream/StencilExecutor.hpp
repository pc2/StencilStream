/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "ExecutionKernel.hpp"
#include "Grid.hpp"
#include "RuntimeSample.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

namespace stencil
{
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1, uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t burst_size = 1024>
class StencilExecutor
{
public:
    static_assert(burst_size % sizeof(T) == 0);
    static constexpr uindex_t burst_length = burst_size / sizeof(T);
    static constexpr uindex_t halo_radius = stencil_radius * pipeline_length;

    StencilExecutor(cl::sycl::buffer<T, 2> input_buffer, T halo_value, TransFunc trans_func) : input_grid(input_buffer, halo_value), queue(), trans_func(trans_func), i_generation(0), runtime_analysis_enabled(false)
    {
    }

    StencilExecutor(uindex_t grid_width, uindex_t grid_height, T halo_value, TransFunc trans_func) : input_grid(grid_width, grid_height, halo_value), queue(), trans_func(trans_func), i_generation(0), runtime_analysis_enabled(false)
    {
    }

    void run(uindex_t n_generations)
    {
        using in_pipe = cl::sycl::pipe<class in_pipe_id, T>;
        using out_pipe = cl::sycl::pipe<class out_pipe_id, T>;
        using ExecutionKernelImpl = ExecutionKernel<TransFunc, T, stencil_radius, pipeline_length, tile_width, tile_height, in_pipe, out_pipe>;

        if (!this->queue.has_value())
        {
            select_emulator(false);
        }

        cl::sycl::queue queue = *(this->queue);

        uindex_t n_passes = n_generations / pipeline_length;
        if (n_generations % pipeline_length != 0)
        {
            n_passes += 1;
        }

        if (runtime_analysis_enabled)
        {
            runtime_sample = RuntimeSample(n_passes, input_grid.get_tile_range().c, input_grid.get_tile_range().r);
        }

        uindex_t grid_width = input_grid.get_grid_range().c;
        uindex_t grid_height = input_grid.get_grid_range().r;
        T default_value = input_grid.get_default_value();

        for (uindex_t i = 0; i < n_passes; i++)
        {
            Grid output_grid = input_grid.make_output_grid();
            uindex_t n_generations_per_pass = pipeline_length;
            if (i == n_passes - 1 && n_generations % pipeline_length != 0)
            {
                n_generations_per_pass = n_generations % pipeline_length;
            }

            for (uindex_t c = 0; c < input_grid.get_tile_range().c; c++)
            {
                for (uindex_t r = 0; r < input_grid.get_tile_range().r; r++)
                {
                    input_grid.template submit_tile_input<in_pipe>(queue, UID(c, r));

                    cl::sycl::event computation_event = queue.submit([&](cl::sycl::handler &cgh) {
                        cgh.single_task(ExecutionKernelImpl(
                            trans_func,
                            i_generation,
                            n_generations_per_pass,
                            c * tile_width,
                            r * tile_height,
                            grid_width,
                            grid_height,
                            default_value));
                    });

                    if (runtime_analysis_enabled)
                    {
                        runtime_sample->add_event(computation_event, i, c, r);
                    }

                    output_grid.template submit_tile_output<out_pipe>(queue, UID(c, r));
                }
            }
            input_grid = output_grid;
            i_generation += n_generations_per_pass;
        }
    }

    void set_input(cl::sycl::buffer<T, 2> input_buffer, T halo_value)
    {
        this->input_grid = GridImpl(input_buffer, halo_value);
    }

    void copy_output(cl::sycl::buffer<T, 2> output_buffer)
    {
        input_grid.copy_to(output_buffer);
    }

    void set_queue(cl::sycl::queue queue, bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = queue;
        verify_queue_properties();
    }

    void select_emulator()
    {
        select_emulator(false);
    }

    void select_fpga()
    {
        select_fpga(false);
    }

    void select_emulator(bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_emulator_selector(), get_queue_properties());
    }

    void select_fpga(bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_selector(), get_queue_properties());
    }

    void set_trans_func(TransFunc trans_func)
    {
        this->trans_func = trans_func;
    }

    uindex_t get_i_generation() const
    {
        return i_generation;
    }

    void set_i_generation(uindex_t i_generation)
    {
        this->i_generation = i_generation;
    }

    std::optional<RuntimeSample> get_runtime_sample()
    {
        return runtime_sample;
    }

private:
    cl::sycl::property_list get_queue_properties()
    {
        cl::sycl::property_list properties;
        if (runtime_analysis_enabled)
        {
            properties = {cl::sycl::property::queue::enable_profiling{}};
        }
        else
        {
            properties = {};
        }
        return properties;
    }

    void verify_queue_properties()
    {
        if (!queue.has_value())
            return;

        if (runtime_analysis_enabled && !queue->has_property<cl::sycl::property::queue::enable_profiling>())
        {
            throw std::runtime_error("Runtime analysis is enabled, but the queue does not support it.");
        }
    }

    using GridImpl = Grid<T, tile_width, tile_height, halo_radius, burst_length>;

    GridImpl input_grid;
    std::optional<cl::sycl::queue> queue;
    TransFunc trans_func;
    uindex_t i_generation;
    bool runtime_analysis_enabled;
    std::optional<RuntimeSample> runtime_sample;
};
} // namespace stencil