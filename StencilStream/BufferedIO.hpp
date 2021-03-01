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
#include "GenericID.hpp"
#include <CL/sycl/accessor.hpp>

namespace stencil_stream
{

template <typename T, UIndex flush_length, cl::sycl::access::mode AccessMode, cl::sycl::access::target AccessTarget=cl::sycl::access::target::global_buffer>
class BufferedIO
{
public:
    using Accessor = cl::sycl::accessor<T, 2, AccessMode, AccessTarget>;

    BufferedIO(bool input_mode, Accessor accessor) : input_mode(input_mode), accessor(accessor), block_i(0), n_blocks(accessor.get_range()[0]), cache(), active_bank(0), cell_i(0)
    {
        assert(accessor.get_range()[1] == flush_length);
        assert(n_blocks >= 2);

        if (input_mode)
        {
#pragma unroll
            for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
            {
                cache[0][cell_i] = accessor[0][cell_i];
                cache[1][cell_i] = accessor[1][cell_i];
            }
            block_i = 2;
        }
    }

    T read()
    {
        T new_value = cache[active_bank][cell_i];

        if (input_mode)
        {
            step_indices([this]() {
#pragma unroll
                for (UIndex load_i = 0; load_i < flush_length; load_i++)
                {
                    cache[this->passive_bank()][load_i] = accessor[this->block_i][load_i];
                }
            });
        }

        return new_value;
    }

    void write(T new_value)
    {
        if (!input_mode)
        {
            cache[active_bank][cell_i] = new_value;

            step_indices([this]() {
#pragma unroll
                for (UIndex store_i = 0; store_i < flush_length; store_i++)
                {
                    accessor[this->block_i][store_i] = cache[this->passive_bank()][store_i];
                }
            });
        }
    }

    void flush()
    {
        for (UIndex store_i = 0; store_i < cell_i; store_i++)
        {
            if (!input_mode)
            {
                accessor[block_i][store_i] = cache[active_bank][store_i];
            }
        }
    }

private:
    UIndex passive_bank() const
    {
        return active_bank == 0 ? 1 : 0;
    }

    template <typename TransferOperation>
    void step_indices(TransferOperation trans_op)
    {
        static_assert(std::is_invocable<TransferOperation>::value);

        // If all cells in the active cache block have been filled,
        if (cell_i == flush_length - 1)
        {
            // swap the banks, read from the previously passive bank.
            active_bank = passive_bank();
            cell_i = 0;

            // If there is one unwritten block,
            if (block_i < n_blocks)
            {
                trans_op();
                block_i++;
            }
        }
        else
        {
            cell_i++;
        }
    }

    bool input_mode;

    Accessor accessor;
    UIndex block_i;
    UIndex n_blocks;

    [[intel::fpga_memory, intel::numbanks(2)]] T cache[2][flush_length];
    UIndex active_bank;
    UIndex cell_i;
};

} // namespace stencil_stream