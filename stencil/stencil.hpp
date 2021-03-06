/**
 * StencilStream - Generic Stencil Simulation Library for FPGAs.
 * 
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * # Conventions
 * 
 * If not stated otherwise, the origin of a buffer is in the top left corner and the first coordinate is the column and the second coordinate is the row of the indexed element. This means that buffers are iterated from left to right and from top to bottom.
 *
 * Concerning this implementation, a stencil buffer is a two-dimensional square lattice, defined by the datatype of it's elements and it's radius (the maximal Chebyshev distance from the center to the edge of the stencil buffer). The origin of the stencil buffer is always at (0,0) and it ranges from (-r,-r) to (r,r), where r is the radius of the stencil. This also means that a stencil buffer is always 2*r + 1 by 2*r + 1 elements in size.
 * 
 * A stencil kernel is a callable object that takes a constant reference to a stencil as well as an info struct and returns a new value. Given a datatype T and a radius, the exact signature is
 *
 * `T kernel(stencil::Stencil2D<T, radius> const&, stencil::Stencil2DInfo &const);`
 * 
 * # Preprocessor parameters
 * 
 * The general goal is to implement as many parameters as template parameters of `StencilExecutor` as possible, but for certain parameters, this isn't possible:
 * * -DSTENCIL_INDEX_WIDTH: The width of the integer types for indexing, defaults to 64.
 * Decreasing the width of the index type might lower the resource usage, but also limits the size of the processable matrices. Static asserts throughout the code ensure that the index type is wide enough.
 * * -DSTENCIL_PIPELINE_LEN: The number of kernels to execute in parallel, defaults to 1.
 * A longer pipeline leads to more parallelity and therefore speed, but also to more resource usage, longer synthesis times and potentially lower clocking frequency.
 */
#pragma once
#include "exec.hpp"
#include "io.hpp"

namespace stencil
{

/**
 * Stencil kernel executor.
 * 
 * This class is used to configure the stencil kernel for execution.
 * 
 * Template Arguments:
 * * typename T: The cell type used by the stencil kernel.
 * * uindex_t radius: The radius of the stencil buffer.
 * * uindex_t max_grid_height: The maximal height of the cell buffer.
 *     This has to be fixed since fixed-sized caches are used internally.
 * * uindex_t burst_size=1024: The size of a global memory burst. 
 *     Memory accesses are arranged in blocks, one block is loaded or stored by one operation.
 *     Depending on the FPGA, certain burst sizes provide a better performance than others.
 *     This parameter can be set to account for this.
 */
template <typename T, uindex_t radius, uindex_t max_grid_width, uindex_t max_grid_height, uindex_t burst_size = 1024>
class StencilExecutor
{
    // Assert that the pipeline length does not exceed the number of pre-generated macros.
    static_assert(STENCIL_PIPELINE_LEN <= STENCIL_MAX_PIPELINE_LEN);
    // Assert that no cells are split between bursts.
    static_assert(burst_size % sizeof(T) == 0);

public:
    // The number of cells in one burst block.
    static constexpr uindex_t block_size = burst_size / sizeof(T);
    // Assert that the working buffer can be partitioned into two halfs and blocks.
    static_assert((max_grid_width * max_grid_height) % (2 * block_size) == 0);

    // The number of blocks per buffer.
    static constexpr uindex_t n_blocks = (max_grid_width * max_grid_height) / block_size;

    /**
     * Create a new stencil executor.
     */
    StencilExecutor(cl::sycl::queue queue)
        : orig_buffer(cl::sycl::range<2>(block_size, 2)), n_generations(1), queue(queue)
    {
    }

    /**
     * Set the buffer to work on.
     * 
     * The results will be written back to this exact buffer.
     * 
     * Since the maximal buffer height is hard-coded into the design via a template
     * argument, the cell buffer height may not exeed this limit.
     */
    void set_buffer(cl::sycl::buffer<T, 2> buffer)
    {
        if (buffer.get_range()[0] > max_grid_width)
        {
            std::cerr << "The buffer is too wide." << std::endl;
            throw std::exception();
        }

        if (buffer.get_range()[1] > max_grid_height)
        {
            std::cerr << "The buffer is too high." << std::endl;
            throw std::exception();
        }

        this->orig_buffer = buffer;
    }

    /**
     * Return the buffer used by the executor.
     */
    cl::sycl::buffer<T, 2> get_buffer() const
    {
        return orig_buffer;
    }

    /**
     * Set the number of generations to compute.
     * 
     * Due to obvious reasons, this number must not be zero. If the number of generations isn't a
     * multiple of the pipeline length, the stencil kernel will still be executed for the missing
     * generations. Therefore, none of the stencil kernel executions, except for the first one, must
     * have side-effects.
     */
    void set_generations(uindex_t new_n_generations)
    {
        if (n_generations == 0)
        {
            std::cerr << "Can not calculate 0 generations" << std::endl;
            throw std::exception();
        }
        n_generations = new_n_generations;
    }

    /**
     * Apply the stencil kernel.
     * 
     * This method enqueues the stencil kernel for execution and waits until it's complete. The SYCL
     * event of the stencil kernel is returned for post-execution analysis (like enqueued time,
     * overall runtime, etc.).
     */
    template <typename Kernel>
    cl::sycl::event run(Kernel kernel)
    {
        static_assert(
            std::is_invocable_r<T, Kernel, Stencil2D<T, radius> const &, Stencil2DInfo const &>::
                value);

        using ExecutionKernel = ExecutionKernel<T, radius, max_grid_width, max_grid_height, block_size, Kernel>;

        // Calculate the number of grid passes to execute.
        uindex_t n_passes = n_generations / pipeline_length;
        if (n_generations % pipeline_length > 0)
        {
            n_passes += 1;
        }

        // Submit the working kernel.
        cl::sycl::event work_event = queue.submit([&](cl::sycl::handler &cgh) {
            ExecutionKernel exec_kernel(n_generations, n_passes, kernel);

            cgh.single_task<class ExecKernelID>(exec_kernel);
        });

        cl::sycl::buffer<T, 2> in_buffer = make_in();

        cl::sycl::buffer<T, 2> out_buffer = feed_grid_to_worker<ExecutionKernel>(in_buffer, n_passes);

        work_event.wait();

        write_back(out_buffer);

        return work_event;
    }

private:
    cl::sycl::buffer<T, 2> make_in()
    {
        uindex_t grid_width = orig_buffer.get_range()[0];
        uindex_t grid_height = orig_buffer.get_range()[1];
        cl::sycl::buffer<T, 2> in_buffer(cl::sycl::range<2>(max_grid_width, max_grid_height));

        auto orig_buffer_ac = orig_buffer.template get_access<cl::sycl::access::mode::read>();
        auto in_buffer_ac = in_buffer.template get_access<cl::sycl::access::mode::discard_write>();

        for (uindex_t c = 0; c < grid_width; c++)
        {
            for (uindex_t r = 0; r < grid_height; r++)
            {
                in_buffer_ac[c][r] = orig_buffer_ac[c][r];
            }
        }

        in_buffer = in_buffer.template reinterpret<T, 2>(cl::sycl::range<2>(max_grid_width * max_grid_height / block_size, block_size));
        return in_buffer;
    }

    /**
     * Manager for a single half-buffer.
     * 
     * It is used to either create or derive the half-buffer, submit the kernel to the queue, synchronize it to other kernel submissions, and wait for it's completion.
     */
    class BufferEventGroup
    {
        std::deque<cl::sycl::event> events;
        std::deque<sync_buffer> sync_buffers;
        cl::sycl::buffer<T, 2> half_buffer;

    public:
        /**
         * Create a group with an anonymous half-buffer.
         * 
         * The new half-buffer isn't accessible outside of this group and only used to store intermediate values.
         */
        BufferEventGroup() : events(), sync_buffers(), half_buffer(cl::sycl::range<2>(n_blocks / 2, block_size)) {}

        /**
         * Create a group with a derived half-buffer.
         * 
         * The new half-buffer is either the upper or the lower half of the given, full buffer, depending on `half`. If `half` is 0, the upper half of the buffer is used, if `half` is 1, the lower half of the buffer is used.
         */
        BufferEventGroup(cl::sycl::buffer<T, 2> buffer, uindex_t half) : events(), sync_buffers(), half_buffer(buffer, cl::sycl::id<2>(half * n_blocks / 2, 0), cl::sycl::range<2>(n_blocks / 2, block_size))
        {
            assert(half == 1 || half == 0);
        }

        /**
         * Submit the IO kernel to the queue.
         * 
         * This will use the previously created or derive half-buffer for operation and synchronize it with the given sync buffer.
         * 
         * The sync buffer will be replaced with a new one. Using the new sync buffer for further submissions assures that these following submissions will be executed after this one.
         */
        template <typename IOKernel>
        void submit(cl::sycl::queue queue, sync_buffer &sync)
        {
            sync_buffer next_sync(cl::sycl::range<1>(1));
            events.push_front(IOKernel::submit(queue, half_buffer, sync, next_sync));
            sync_buffers.push_front(sync);
            sync = next_sync;
        }

        /**
         * Wait for all submissions to finish. 
         */
        void wait()
        {
            while (!events.empty())
            {
                events.back().wait();
                events.pop_back();
            }
        }
    };

    /**
     * Manager for all enqueued kernel submissions.
     */
    class GroupQueue
    {
        std::deque<BufferEventGroup> groups;

        /**
         * Return the newest, latest group, or none if it doesn't exist.
         */
        std::optional<BufferEventGroup> latest_group()
        {
            if (groups.empty())
            {
                return std::nullopt;
            }
            else
            {
                return groups.front();
            }
        }

    public:
        /**
         * Create a new group queue.
         */
        GroupQueue() : groups() {}

        /**
         * Create a new group with an anonymous buffer.
         * 
         * See BufferEventGroup() for details.
         */
        void new_group()
        {
            groups.push_front(BufferEventGroup());
        }

        /**
         * Create a new group with access to one half of the buffer.
         * 
         * See BufferEventGroup(cl::sycl::buffer<T, 2>, uindex_t) for details.
         */
        void new_group(cl::sycl::buffer<T, 2> buffer, uindex_t half)
        {
            groups.push_front(BufferEventGroup(buffer, half));
        }

        /**
         * Wait for all submissions to complete and de-allocate their buffers.
         */
        void wait()
        {
            while (!groups.empty())
            {
                groups.back().wait();
                groups.pop_back();
            }
        }

        /**
         * Submit a new IO kernel invocation.
         * 
         * See BufferEventGroup::submit for details. Fails if no group has been created by throwing std::bad_optional_access.
         */
        template <typename IOKernel>
        void submit(cl::sycl::queue queue, sync_buffer &sync)
        {
            BufferEventGroup latest_group = this->latest_group().value();
            try
            {
                // Try to submit the kernel invocation.
                latest_group.template submit<IOKernel>(queue, sync);
            }
            catch (cl::sycl::nd_range_error e)
            {
                // If the buffer allocation fails, wait for all previous invocations and deallocate
                // their buffers.
                wait();
                // The latest group has been removed from the queue and has to be added again.
                groups.push_front(latest_group);
                // Submit the kernel invocation. There should be space for it now.
                latest_group.template submit<IOKernel>(queue, sync);
            }
        }
    };

    template <typename ExecutionKernel>
    cl::sycl::buffer<T, 2> feed_grid_to_worker(cl::sycl::buffer<T, 2> in_buffer, uindex_t n_passes)
    {
        assert(in_buffer.get_range()[0] == n_blocks);
        assert(in_buffer.get_range()[1] == block_size);

        using InputKernel = IOKernel<T, n_blocks / 2, block_size, true, ExecutionKernel>;
        using OutputKernel = IOKernel<T, n_blocks / 2, block_size, false, ExecutionKernel>;

        GroupQueue groups;

        sync_buffer input_sync(cl::sycl::range<1>(1));
        sync_buffer output_sync(cl::sycl::range<1>(1));

        // Submit the InputKernel with one half of the in_buffer each.
        for (uindex_t half = 0; half < 2; half++)
        {
            groups.new_group(in_buffer, half);
            groups.template submit<InputKernel>(queue, input_sync);
        }

        // Feed the output of the execution kernel back to it for every additional grid pass.
        for (uindex_t half = 0; half < 2 * (n_passes - 1); half++)
        {
            groups.new_group();
            groups.template submit<OutputKernel>(queue, output_sync);
            groups.template submit<InputKernel>(queue, input_sync);
        }

        // Collect the final output in a separate buffer.
        cl::sycl::buffer<T, 2> out_buffer(cl::sycl::range<2>(n_blocks, block_size));

        for (uindex_t half = 0; half < 2; half++)
        {
            groups.new_group(out_buffer, half);
            groups.template submit<OutputKernel>(queue, output_sync);
        }

        // Wait for all kernels to finish and deallocate their buffers in the process.
        groups.wait();

        return out_buffer;
    }

    void write_back(cl::sycl::buffer<T, 2> out_buffer)
    {
        uindex_t grid_width = orig_buffer.get_range()[0];
        uindex_t grid_height = orig_buffer.get_range()[1];

        out_buffer = out_buffer.template reinterpret<T, 2>(cl::sycl::range<2>(max_grid_width, max_grid_height));

        auto out_buffer_ac = out_buffer.template get_access<cl::sycl::access::mode::read>();
        auto orig_buffer_ac = orig_buffer.template get_access<cl::sycl::access::mode::discard_write>();

        for (uindex_t c = 0; c < grid_width; c++)
        {
            for (uindex_t r = 0; r < grid_height; r++)
            {
                orig_buffer_ac[c][r] = out_buffer_ac[c][r];
            }
        }
    }

    cl::sycl::buffer<T, 2> orig_buffer;
    uindex_t n_generations;
    cl::sycl::queue queue;
};
} // namespace stencil
