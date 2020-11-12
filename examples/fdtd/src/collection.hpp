#pragma once
#include "defines.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <sstream>
#include <thread>
#include <vector>

class SampleCollectorKernel
{
    accessor<float_vec, 2, access::mode::write> samples;

public:
    using in_pipe = cl::sycl::pipe<class in_pipe_id, float_vec, frame_block_size>;

    SampleCollectorKernel(buffer<float_vec, 2> samples_buffer, handler &cgh)
        : samples(samples_buffer.get_access<access::mode::write>(cgh))
    {
        assert(samples_buffer.get_range()[0] == blocks_per_frame);
        assert(samples_buffer.get_range()[1] == frame_block_size);
    }

    void operator()()
    {
        for (uindex_t block_i = 0; block_i < blocks_per_frame; block_i++)
        {
            float_vec block[frame_block_size];

            for (uindex_t value_i = 0; value_i < frame_block_size; value_i++)
            {
                block[value_i] = in_pipe::read();
            }

#pragma unroll
            for (uindex_t value_i = 0; value_i < frame_block_size; value_i++)
            {
                samples[block_i][value_i] = block[value_i];
            }
        }
    }
};

class SampleCollector
{
    cl::sycl::event collection_event;
    cl::sycl::buffer<float_vec, 2> frame_buffer;
    uindex_t frame_index;
    uindex_t n_frames;
    bool write_frames;

    SampleCollector(cl::sycl::queue device_queue, uindex_t frame_index, uindex_t n_frames, bool write_frames) : collection_event(), frame_buffer(range<2>(blocks_per_frame, frame_block_size)), frame_index(frame_index), n_frames(n_frames), write_frames(write_frames)
    {
        collection_event = device_queue.submit([&](handler &cgh) {
            cgh.single_task<class SampleCollectorKernel>(SampleCollectorKernel(frame_buffer, cgh));
        });
    }

public:
    void operator()()
    {
        collection_event.wait();

        auto samples = frame_buffer.get_access<access::mode::read>();

        cout << "Collected frame " << frame_index + 1 << "/" << n_frames << std::endl;

        if (write_frames)
        {
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
                        out << samples[b][i][j] << std::endl;
                    }
                }
            }
        }

        cout << "Written frame " << frame_index + 1 << "/" << n_frames << std::endl;
    }

    static std::deque<std::thread> start_collecting(cl::sycl::queue device_queue, Parameters const &parameters)
    {
        std::deque<std::thread> collector_threads;
        for (uindex_t i = 0; i < parameters.n_frames(); i++)
        {
            SampleCollector collector(device_queue, i, parameters.n_frames(), parameters.write_frames);
            collector_threads.push_front(std::thread(collector));
        }
        return collector_threads;
    }
};