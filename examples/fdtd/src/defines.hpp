/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the “Software”), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/Index.hpp>
#include <fstream>
#include <iostream>
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

// Square root of two
constexpr float sqrt_2 = 1.4142135623730951;

/* stencil parameters */
constexpr uindex_t tile_height = 512;
constexpr uindex_t tile_width = 512;
constexpr uindex_t stencil_radius = 1;

#ifdef HARDWARE
constexpr uindex_t pipeline_length = 128;
#else
constexpr uindex_t pipeline_length = 16;
#endif

std::string description = "\
This application simulates a nano-photonic disk cavity.\n\
\n\
The simulated cavity is a two-dimensional circular cavity in perfectly conducting metal. An inital\n\
magnetic source wave is applied, which is cut off after a certain amount of time. Then, after\n\
a certain amount of time has passed, the resulting magnetic field value is accumulated and written\n\
to the CSV file \"hz_sum.csv\".\n\
\n\
In the verbose or debugging mode, the application prints the progress of the simulation roughly once\n\
per second and writes the current state of the magnetic field to \"hz.<time>.csv\", where <time> is\n\
the passed simulation time measured in multiples of tau.\n\
\n\
The CSV files use commata (\",\") as field separators and new-lines (\"\\n\") as line separators.\n\
\n\
Simulation Parameters:\n\
-c <float>: Time instant t_cutoff when the source wave is cut off, measured in multiples of tau (default: 7.0 tau).\n\
-d <float>: Time instant t_detect when the field accumulation starts, measured in multiples of tau (default: 14 tau).\n\
-e <float>: Time instant t_max when the simulation stops, measured in multiples of tau (default: 15 tau).\n\
-f <float>: The frequency of the source wave, measured in Hz (default: 120e12 Hz).\n\
-p <float>: Phase of the source wave, measured in multiples of tau (default: 3.0 tau).\n\
-r <float>: Radius of the cavity, measured in meters (default: 800e-9 m).\n\
-s <float>: The spatial resultion, measured in meters per grid cell (default: 10e-9 m/cell).\n\
-t <float>: Timescale tau for the simulation, measured in in seconds (default: 100e-15 s).\n\
\n\
Operation Parameters:\n\
-h:         Print this help message and exit.\n\
-o <path>:  Directory for output files (default: \".\").\n\
-i <float>: Write a snapshot of the magnetic field to the output directory every x multiples of tau (default: disabled).\n\
";

struct FDTDCell {
    float ex, ey, hz, hz_sum, distance, pad0, pad1, pad2;
};

// The coefficients that describe the properties of a material.
struct Material {
    float ca;
    float cb;
    float da;
    float db;
};

struct Parameters {
    Parameters(int argc, char **argv)
        : t_cutoff_factor(7.0), t_detect_factor(14.0), t_max_factor(15.0), frequency(120e12),
          t_0_factor(3.0), disk_radius(800e-9), dx(10e-9), tau(100e-15), out_dir("."),
          interval_factor(std::nullopt) {
        int c;
        while ((c = getopt(argc, argv, "hc:d:e:f:p:r:s:t:o:i:")) != -1) {
            switch (c) {
            case 'c':
                t_cutoff_factor = stof(optarg);
                if (t_cutoff_factor < 0.0) {
                    cerr << "Error: t_cutoff may not be negative!" << std::endl;
                    exit(1);
                }
                break;
            case 'd':
                t_detect_factor = stof(optarg);
                if (t_detect_factor < 0.0) {
                    cerr << "Error: t_detect may not be negative!" << std::endl;
                    exit(1);
                }
                break;
            case 'e':
                t_max_factor = stof(optarg);
                if (t_max_factor < 0.0) {
                    cerr << "Error: t_max may not not be negative!" << std::endl;
                    exit(1);
                }
                break;
            case 'f':
                frequency = stof(optarg);
                if (frequency < 0.0) {
                    cerr << "Error: The frequency of the source wave may not be negative"
                         << std::endl;
                    exit(1);
                }
                break;
            case 'p':
                t_0_factor = stof(optarg);
                if (t_0_factor < 0.0) {
                    cerr << "Error: The phase of the source wave may not be negative" << std::endl;
                    exit(1);
                }
                break;
            case 'r':
                disk_radius = stof(optarg);
                if (disk_radius < 0.0) {
                    cerr << "Error: The disk radius wave may not be negative" << std::endl;
                    exit(1);
                }
                break;
            case 's':
                dx = stof(optarg);
                if (dx < 0.0) {
                    cerr << "Error: The spatial resolution may not be negative" << std::endl;
                    exit(1);
                }
                break;
            case 't':
                tau = stof(optarg);
                if (tau < 0.0) {
                    cerr << "Error: Tau may not be negative" << std::endl;
                    exit(1);
                }
                break;
            case 'o':
                out_dir = std::string(optarg);
                break;
            case 'i':
                interval_factor = stof(optarg);
                break;
            case 'h':
            case '?':
            default:
                cerr << description;
                exit(1);
            }
        }
    }

    float t_cutoff_factor;

    float t_detect_factor;

    float t_max_factor;

    float frequency;

    float t_0_factor;

    float disk_radius;

    float dx;

    float tau;

    std::string out_dir;

    std::optional<float> interval_factor;

    float t_cutoff() const { return t_cutoff_factor * tau; }

    float t_detect() const { return t_detect_factor * tau; }

    float t_max() const { return t_max_factor * tau; }

    float t_0() const { return t_0_factor * tau; }

    float dt() const { return (dx / float(c0 * sqrt_2)) * 0.99; }

    uindex_t n_timesteps() const { return 2 * uindex_t(std::ceil(t_max() / dt())); }

    // Omega (?) in Hz.
    float omega() const { return 2.0 * M_PI * frequency; }

    cl::sycl::range<2> grid_range() const {
        uindex_t width = uindex_t(std::ceil((2 * disk_radius / dx) + 2));
        uindex_t height = width;
        return cl::sycl::range<2>(width, height);
    }

    Material vacuum() const {
        const Material vacuum{
            (1 - (sigma * dt()) / (2 * eps_0 * eps_r)) /
                (1 + (sigma * dt()) / (2 * eps_0 * eps_r)),                             // ca
            (dt() / (eps_0 * eps_r * dx)) / (1 + (sigma * dt()) / (2 * eps_0 * eps_r)), // cb
            (1 - (sigma * dt()) / (2 * mu_0)) / (1 + (sigma * dt()) / (2 * mu_0)),      // da
            (dt() / (mu_0 * dx)) / (1 + (sigma * dt()) / (2 * mu_0)),                   // db
        };
        return vacuum;
    }

    std::optional<uindex_t> interval() const {
        if (interval_factor.has_value()) {
            return uindex_t(std::ceil((*interval_factor * tau) / dt()));
        } else {
            return std::nullopt;
        }
    }

    void print_configuration() const {
        std::cout << "Simulation Configuration:" << std::endl;
        std::cout << std::endl;
        std::cout << "# Timing" << std::endl;
        std::cout << "tau           = " << tau << " s" << std::endl;
        std::cout << "t_cutoff      = " << t_cutoff_factor << " tau = " << t_cutoff() << " s"
                  << std::endl;
        std::cout << "t_detect      = " << t_detect_factor << " tau = " << t_detect() << " s"
                  << std::endl;
        std::cout << "t_max         = " << t_max_factor << " tau = " << t_max() << " s"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "# Source Wave" << std::endl;
        std::cout << "phase         = " << t_0_factor << " tau = " << t_0() << " s" << std::endl;
        std::cout << "frequency     = " << frequency << " Hz" << std::endl;
        std::cout << std::endl;
        std::cout << "# Vacuum coefficients" << std::endl;
        std::cout << "ca            = " << vacuum().ca << std::endl;
        std::cout << "cb            = " << vacuum().cb << std::endl;
        std::cout << "da            = " << vacuum().da << std::endl;
        std::cout << "db            = " << vacuum().db << std::endl;
        std::cout << std::endl;
        std::cout << "# Execution parameters" << std::endl;
        std::cout << "dx            = " << dx << " m/cell" << std::endl;
        std::cout << "dt            = " << dt() << " s/generation" << std::endl;
        std::cout << "radius        = " << disk_radius << " m" << std::endl;
        std::cout << "grid w/h      = " << grid_range()[0] << " cells" << std::endl;
        std::cout << "n. timesteps  = " << n_timesteps() << std::endl;
        std::cout << std::endl;
    }
};