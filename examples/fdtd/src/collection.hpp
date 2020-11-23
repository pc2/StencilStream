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
#include "defines.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
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
        frame_path << "frame." << frame_index << ".csv.gz";

        boost::iostreams::filtering_ostream out;
        out.push(boost::iostreams::basic_gzip_compressor());
        out.push(boost::iostreams::basic_file_sink<char>(frame_path.str()));

        for (uindex_t b = 0; b < samples.get_range()[0]; b++)
        {
            for (uindex_t i = 0; i < samples.get_range()[1]; i++)
            {
                for (uindex_t j = 0; j < vector_len; j++)
                {
                    out << samples[b][i].hz_sum[j] << std::endl;
                }
            }
        }

        cout << "Written frame " << frame_index << std::endl;
    }
};