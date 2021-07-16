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
#include "defines.hpp"
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

class SampleCollector
{
    uindex_t frame_index;
    cl::sycl::buffer<FDTDCell, 2> frame_buffer;

public:
    SampleCollector(uindex_t frame_index, cl::sycl::buffer<FDTDCell, 2> frame_buffer) : frame_index(frame_index), frame_buffer(frame_buffer)
    {
    }

    void operator()()
    {
        auto samples = frame_buffer.get_access<access::mode::read>();

        ostringstream frame_path;
        frame_path << "frame." << frame_index << ".csv";

        std::ofstream out(frame_path.str());

        for (uindex_t r_v = 0; r_v < samples.get_range()[1]; r_v++)
        {
            for (uindex_t r_i = 0; r_i < vector_len && (r_v * vector_len + r_i) < samples.get_range()[0]; r_i++)
            {
                for (uindex_t c = 0; c < samples.get_range()[0]; c++)
                {
                    out << samples[c][r_v].hz_sum[r_i] << ",";
                }
                out << std::endl;
            }
        }

        cout << "Written frame " << frame_index << std::endl;
    }
};