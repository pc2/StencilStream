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
/**
 * \brief Execution coordinator.
 * 
 * The `StencilExecutor` binds the different parts of StencilStream together and provides a unified
 * interface for applications. Users create a stencil executor, configure it and then use it to run
 * the payload computations. It has multiple logical attributes that can be configured:
 * 
 * ### Grid 
 * 
 * The grid is the logical array of cells. It can be initialized either from a given height and
 * width or from a SYCL buffer. A stencil executor does not work in place and a buffer used to
 * initialize the grid can be used for other tasks afterwards. The \ref StencilExecutor.run method alters the state of
 * the grid and the grid can be copied back to a given buffer using \ref StencilExecutor.copy_output.
 * 
 * ### SYCL queue
 * 
 * The queue provides the OpenCL platform, device, context and queue to execute the kernels. A stencil
 * executor does not have a queue when constructed and \ref StencilExecutor.run tries to select the
 * FPGA emulation device if no queue has been configured yet. \ref StencilExecutor.set_queue is used
 * to directly set the queue, but \ref StencilExecutor.select_emulator and
 * \ref StencilExecutor.select_fpga can be used to automatically configure the FPGA emulator or the
 * FPGA, respectively.
 * 
 * ### Transition Function
 * 
 * A stencil executor stores an instance of the transition function since it may require some
 * configuration and runtime-dynamic parameters too. An instance is required for the initialization,
 * but it may be replaced at any time with \ref StencilExecutor.set_trans_func.
 * 
 * ### Generation Index
 * 
 * This is the generation index of the current state of the grid. \ref StencilExecutor.run updates and therefore, it can be ignored in most instances. However, it can be reset if a transition function needs it.
 * 
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the central cell. Must be at least 1.
 * \tparam TransFunc An invocable type that maps a \ref Stencil to the next generation of the stencil's central cell.
 * \tparam pipeline_length The number of hardware execution stages. Must be at least 1. Defaults to 1.
 * \tparam tile_width The number of columns in a tile. Defaults to 1024.
 * \tparam tile_height The number of rows in a tile. Defaults to 1024.
 * \tparam burst_size The number of bytes to load/store in one burst. Defaults to 1024.
 */
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1, uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t burst_size = 1024>
class StencilExecutor
{
public:
    static_assert(burst_size % sizeof(T) == 0);
    static constexpr uindex_t burst_length = burst_size / sizeof(T);
    static constexpr uindex_t halo_radius = stencil_radius * pipeline_length;

    /**
     * \brief Create a new stencil executor.
     * 
     * \param input_buffer The initial state of the grid.
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    StencilExecutor(cl::sycl::buffer<T, 2> input_buffer, T halo_value, TransFunc trans_func) : input_grid(input_buffer), queue(), trans_func(trans_func), i_generation(0), halo_value(halo_value), runtime_analysis_enabled(false)
    {
    }

    /**
     * \brief Compute the next generations.
     * 
     * It uses the transition function to advance the state of the grid.
     * 
     * \param n_generations The number of generations to advance the state of the grid by.
     */
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

                    cl::sycl::event computation_event = queue.submit([&](cl::sycl::handler &cgh)
                                                                     { cgh.single_task(ExecutionKernelImpl(
                                                                           trans_func,
                                                                           i_generation,
                                                                           n_generations_per_pass,
                                                                           c * tile_width,
                                                                           r * tile_height,
                                                                           grid_width,
                                                                           grid_height,
                                                                           halo_value)); });

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

    /**
     * \brief Set the state of the grid.
     * 
     * \param input_buffer A buffer containing the new state of the grid.
     */
    void set_input(cl::sycl::buffer<T, 2> input_buffer)
    {
        this->input_grid = GridImpl(input_buffer);
    }

    /**
     * \brief Copy the current state of the grid to the buffer.
     * 
     * The \ref output_buffer has to have the exact range as returned by \ref StencilExecutor.get_grid_range.
     * 
     * \param output_buffer Copy the state of the grid to this buffer.
     */
    void copy_output(cl::sycl::buffer<T, 2> output_buffer)
    {
        input_grid.copy_to(output_buffer);
    }

    /**
     * \brief Return the range of the grid.
     * 
     * \return The range of the grid.
     */
    UID get_grid_range() const
    {
        return input_grid.get_grid_range();
    }

    /**
     * \brief Return the value of cells in the grid halo.
     * 
     * \return The value of cells in the grid halo.
     */
    T get_halo_value() const
    {
        return halo_value;
    }

    /**
     * \brief Set the value of cells in the grid halo.
     * 
     * \param halo_value The new value of cells in the grid halo.
     */
    void set_halo_value(T halo_value)
    {
        this->halo_value = halo_value;
    }

    /**
     * \brief Manually set the SYCL queue to use for execution.
     * 
     * Note that as of OneAPI Version 2021.1.1, device code is usuallly built either for CPU/GPU, for the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will lead to exceptions.
     * 
     * In order to use runtime analysis features, the queue has to be configured with the `cl::sycl::property::queue::enable_profiling` property.
     * 
     * \param queue The new SYCL queue to use for execution.
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    void set_queue(cl::sycl::queue queue, bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = queue;
        verify_queue_properties();
    }

    /**
     * \brief Set up a SYCL queue with the FPGA emulator device.
     * 
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will lead to exceptions.
     */
    void select_emulator()
    {
        select_emulator(false);
    }

    /**
     * \brief Set up a SYCL queue with an FPGA device.
     * 
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will lead to exceptions.
     */
    void select_fpga()
    {
        select_fpga(false);
    }

    /**
     * \brief Set up a SYCL queue with the FPGA emulator device and optional runtime analyis.
     * 
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will lead to exceptions.
     * 
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    void select_emulator(bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_emulator_selector(), get_queue_properties());
    }

    /**
     * \brief Set up a SYCL queue with an FPGA device and optional runtime analyis.
     * 
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will lead to exceptions.
     * 
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    void select_fpga(bool runtime_analysis)
    {
        runtime_analysis_enabled = runtime_analysis;
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_selector(), get_queue_properties());
    }

    /**
     * \brief Update the transition function instance.
     * 
     * \param trans_func The new transition function instance.
     */
    void set_trans_func(TransFunc trans_func)
    {
        this->trans_func = trans_func;
    }

    /**
     * \brief Get the current generation index of the grid.
     * 
     * \return The current generation index of the grid.
     */
    uindex_t get_i_generation() const
    {
        return i_generation;
    }

    /**
     * \brief Set the generation index of the grid.
     * 
     * \param i_generation The new generation index of the grid.
     */
    void set_i_generation(uindex_t i_generation)
    {
        this->i_generation = i_generation;
    }

    /**
     * \brief Return the runtime information collected from the last \ref StencilExecutor.run call.
     * 
     * \return The collected runtime information. May be `nullopt` if no runtime analysis was configured.
     */
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
    T halo_value;

    bool runtime_analysis_enabled;
    std::optional<RuntimeSample> runtime_sample;
};
} // namespace stencil