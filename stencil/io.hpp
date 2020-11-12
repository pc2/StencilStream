/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "data.hpp"
#include <queue>

namespace stencil
{

/**
 * The SYCL kernel(s) that send cells from the global buffer to the execution kernel and back.
 * 
 * In order to optimize throughput, the buffer is organized in blocks with a fixed size. One block
 * is read from the global buffer as whole and then sent to the execution kernel value by value.
 * 
 * If input is set to true, this kernel will read cells from the buffer and send them to the execution kernel.
 * If input is set to false, this kernel will receive cells from the execution kernel and write them to the buffer.
 * 
 * The constructor of the kernel is private. In order to submit the kernel, use the static `submit` method.
 */
template <typename T, uindex_t n_blocks, uindex_t block_size, bool input, typename ExecutionKernel>
class IOKernel
{
    cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write> buffer;

    using in_pipe = typename ExecutionKernel::in_pipe;
    using out_pipe = typename ExecutionKernel::out_pipe;

    IOKernel(cl::sycl::buffer<T, 2> buffer, cl::sycl::handler &cgh) : buffer(buffer.template get_access<cl::sycl::access::mode::read_write>(cgh))
    {
        assert(buffer.get_range()[0] == n_blocks);
        assert(buffer.get_range()[1] == block_size);
    }

public:
    void operator()()
    {
        for (uindex_t block_i = 0; block_i < n_blocks; block_i++)
        {
            if (input)
            {
                T block[block_size];

#pragma unroll
                for (uindex_t value_i = 0; value_i < block_size; value_i++)
                {
                    block[value_i] = buffer[block_i][value_i];
                }

                for (uindex_t value_i = 0; value_i < block_size; value_i++)
                {
                    in_pipe::write(block[value_i]);
                }
            }
            else
            {
                T block[block_size];

                for (uindex_t value_i = 0; value_i < block_size; value_i++)
                {
                    block[value_i] = out_pipe::read();
                }

#pragma unroll
                for (uindex_t value_i = 0; value_i < block_size; value_i++)
                {
                    buffer[block_i][value_i] = block[value_i];
                }
            }
        }
    }

    static cl::sycl::event submit(cl::sycl::queue queue, cl::sycl::buffer<T, 2> buffer)
    {
        return queue.submit([&](cl::sycl::handler &cgh) {
            IOKernel kernel(buffer, cgh);
            cgh.single_task<class InputKernel>(kernel);
        });
    }
};

} // namespace stencil