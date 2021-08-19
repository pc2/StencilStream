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
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil
{
template <typename T, uindex_t stencil_radius, typename TransFunc>
class AbstractExecutor
{
public:
    AbstractExecutor(T halo_value, TransFunc trans_func) : halo_value(halo_value), trans_func(trans_func), i_generation(0) {}

    virtual void run(uindex_t n_generations) = 0;

    virtual void set_input(cl::sycl::buffer<T, 2> input_buffer) = 0;

    virtual void copy_output(cl::sycl::buffer<T, 2> output_buffer) = 0;

    T const get_halo_value() const
    {
        return halo_value;
    }

    void set_halo_value(T halo_value)
    {
        this->halo_value = halo_value;
    }

    TransFunc get_trans_func() const
    {
        return trans_func;
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

    void inc_i_generation(index_t delta)
    {
        this->i_generation += delta;
    }

private:
    T halo_value;
    TransFunc trans_func;
    uindex_t i_generation;
};
} // namespace stencil