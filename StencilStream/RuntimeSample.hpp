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
#include "boost/multi_array.hpp"
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/event.hpp>

namespace stencil
{
class RuntimeSample
{
public:
    RuntimeSample(uindex_t n_passes, uindex_t n_tile_columns, uindex_t n_tile_rows) : events(boost::extents[n_passes][n_tile_columns][n_tile_rows]) {}

    void add_event(cl::sycl::event event, uindex_t i_pass, uindex_t i_column, uindex_t i_row)
    {
        events[i_pass][i_column][i_row] = event;
    }

    double get_runtime(uindex_t i_pass, uindex_t i_column, uindex_t i_row)
    {
        unsigned long event_start = events[i_pass][i_column][i_row].get_profiling_info<cl::sycl::info::event_profiling::command_start>();
        unsigned long event_end = events[i_pass][i_column][i_row].get_profiling_info<cl::sycl::info::event_profiling::command_end>();
        return (event_end - event_start) / timesteps_per_second;
    }

    double get_total_runtime()
    {
        unsigned long earliest_start = std::numeric_limits<unsigned long>::max();
        unsigned long latest_end = std::numeric_limits<unsigned long>::min();

        for (uindex_t i_pass = 0; i_pass < events.shape()[0]; i_pass++)
        {
            for (uindex_t i_column = 0; i_column < events.shape()[1]; i_column++)
            {
                for (uindex_t i_row = 0; i_row < events.shape()[2]; i_row++)
                {
                    unsigned long event_start = events[i_pass][i_column][i_row].get_profiling_info<cl::sycl::info::event_profiling::command_start>();
                    unsigned long event_end = events[i_pass][i_column][i_row].get_profiling_info<cl::sycl::info::event_profiling::command_end>();
                    if (event_start < earliest_start)
                        earliest_start = event_start;
                    if (event_end > latest_end)
                        latest_end = event_end;
                }
            }
        }

        return (latest_end - earliest_start) / timesteps_per_second;
    }

private:
    static constexpr double timesteps_per_second = 1000000000.0;

    boost::multi_array<cl::sycl::event, 3> events;
};
} // namespace stencil