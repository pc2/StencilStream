#pragma once
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <fstream>
#include <iostream>
#include <stencil/stencil.hpp>
#include <unistd.h>

using namespace std;
using namespace cl::sycl;
using namespace stencil;

//////////////////////////////////////////////
// Needed physical constants for simulation //
//////////////////////////////////////////////
// velocity of light in m/s
constexpr float c0 = 299792458.0;

// conductivity sigma in S/m
constexpr float sigma = 0.0;

// permeability (transmissibility of the material for magnetic fields) in vacuum
constexpr float mu_0 = 4.0 * M_PI * 1.0e-7;

// permittivity (transmissibility of the material for electric fields) in vacuum
constexpr float eps_0 = 1.0 / (c0 * c0 * mu_0);

// permittivity SI - unit-less
constexpr float eps_r = 11.56;

// one nanometer.
constexpr float nm = 1e-9;

// distance between points in the grid in m
constexpr float dx = 10 * nm;

// Square root of two
constexpr float sqrt_2 = 1.4142135623730951;

// distance between generations in s
constexpr float dt = (dx / (c0 * sqrt_2)) * 0.99;

// Number of values in a vector cell.
constexpr uindex_t vector_len = 2;
typedef cl::sycl::vec<float, vector_len> float_vec;
typedef cl::sycl::vec<double, vector_len> double_vec;

/* stencil parameters */
constexpr uindex_t n_rows = 4096;
constexpr uindex_t n_logical_columns = 4096;
constexpr uindex_t n_columns = n_logical_columns / vector_len;
constexpr stencil::index_t stencil_radius = 1;
const cl::sycl::range<2> working_range(n_columns, n_rows);

// Number of samples in a frame block.
constexpr uindex_t frame_block_size = STENCIL_BURST_SIZE / sizeof(float_vec);
static_assert(n_rows * n_columns % frame_block_size == 0);
constexpr uindex_t blocks_per_frame = n_rows * n_columns / frame_block_size;

// Default radius of the cavity in dx.
constexpr double default_radius = 80.0;
constexpr double default_tau = 100e-15;
constexpr double default_frequency = 121.5e12;

// Coordinates of the cavity center in dx.
constexpr uindex_t mid_y = n_columns / 2;
constexpr uindex_t mid_x = n_rows / 2;

// Collection time for a single frame in dx.
constexpr uindex_t dt_collection = STENCIL_PIPELINE_LEN;

struct Parameters
{
    Parameters(int argc, char **argv) : n_time_steps(pipeline_length), n_sample_steps(pipeline_length), disk_radius(default_radius), write_frames(true)
    {
        int c;
        while ((c = getopt(argc, argv, "sht:c:r:")) != -1)
        {
            switch (c)
            {
            case 't':
                n_time_steps = stol(optarg);
                break;
            case 'c':
                n_sample_steps = stol(optarg);
                if (n_sample_steps % pipeline_length != 0)
                {
                    n_sample_steps += pipeline_length - n_sample_steps % pipeline_length;
                    cerr << "Collecting " << n_time_steps << " time steps per frame to account for pipeline length." << std::endl;
                }
            case 'r':
                disk_radius = stod(optarg);
                break;
            case 's':
                write_frames = false;
                break;
            case 'h':
            case '?':
            default:
                cerr << "Options:" << std::endl;
                cerr << "-t <steps>: Number of time steps to calculate (default " << pipeline_length << ")" << std::endl;
                cerr << "-c <steps>: Number of time steps to collect for a frame (default " << pipeline_length << ")" << std::endl;
                cerr << "-r <radius>: Radius of the cavity in cell widths (default " << default_radius << ")" << std::endl;
                cerr << "-s: Don't write collected frames into files." << std::endl;
                exit(1);
            }
        }
    }

    // The number of dt time steps to calculate.
    uindex_t n_time_steps;

    // The number of samples to collect for one frame.
    // You can think of it as the "shutter speed" of the sample collection.
    uindex_t n_sample_steps;

    // The number of sample frames to collect.
    uindex_t n_frames() const
    {
        assert(n_time_steps % n_sample_steps == 0);
        return n_time_steps / n_sample_steps;
    }

    // Radius of the cavity in dx.
    double disk_radius;

    // Timescale (?) for the source wave in s.
    float tau() const
    {
        return default_tau * (disk_radius / default_radius);
    }

    // The frequency of the source wave in Hz.
    float frequency() const
    {
        return default_frequency * (default_radius / disk_radius);
    }

    // Omega (?) in Hz.
    float omega() const
    {
        return 2.0 * M_PI * frequency();
    }

    // The point in time when the simulation starts in s.
    float t0() const
    {
        return 3.0 * tau();
    }

    // The point in time to cut off the source wave in s.
    float t_cutoff() const
    {
        return 7.0 * tau();
    }

    bool write_frames;
};