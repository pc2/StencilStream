/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "defines.hpp"
#include "material/Material.hpp"
#include <nlohmann/json.hpp>

using namespace nlohmann;

std::string description = R"(
This application simulates a nano-photonic disk cavity.

The simulated cavity is a two-dimensional circular cavity in perfectly conducting metal. An inital
magnetic source wave is applied, which is cut off after a certain amount of time. Then, after
a certain amount of time has passed, the resulting magnetic field value is accumulated and written
to the CSV file "hz_sum.csv".

In the verbose or debugging mode, the application prints the progress of the simulation roughly once
per second and writes the current state of the magnetic field to \"hz.<time>.csv\", where <time> is
the passed simulation time measured in multiples of tau.

The CSV files use commata (",") as field separators and new-lines ("\n") as line separators.

-h:         Print this help message and exit.
-c <path>:  Load the given experiment JSON file. If set to "-", it will be read from stdin (Required).
-o <path>:  Directory for output files (default: ".").
)";

struct Parameters {
    static void print_usage_and_exit(char *process_name) {
        std::cerr << "Usage: " << process_name << " -c <path>" << std::endl;
        std::cerr << description;
        exit(1);
    }

    Parameters(int argc, char **argv)
        : t_cutoff_factor(7.0), t_detect_factor(14.0), t_max_factor(15.0),
          t_snap_factor(std::nullopt), frequency(120e12), t_0_factor(3.0), source_x(0.0),
          source_y(0.0), source_radius(0.0), dx(10e-9), tau(100e-15), rings(), out_dir(".") {

        bool config_loaded = false;

        int c;
        while ((c = getopt(argc, argv, "hc:o:i:")) != -1) {
            if (c == 'h' || c == '?') {
                print_usage_and_exit(argv[0]);
            }

            std::string arg = std::string(optarg);
            index_t first_comma, second_comma;

            switch (c) {
            case 'c':
                load_config(std::string(arg));
                config_loaded = true;
                break;
            case 'o':
                out_dir = std::string(arg);
                break;
            default:
                break;
            }
        }

        if (!config_loaded) {
            print_usage_and_exit(argv[0]);
        }
    }

    static void check_existance(json &object, std::string key) {
        if (!object.contains(key)) {
            std::cerr << "Field '" << key << "' is missing!" << std::endl;
            exit(1);
        }
    }

    static float get_checked_float(json &object, std::string key) {
        check_existance(object, key);
        if (!object[key].is_number()) {
            std::cerr << "Field '" << key << "' has to be a number, but is a "
                      << object[key].type_name() << "!" << std::endl;
            exit(1);
        }
        return object[key].get<float>();
    }

    static index_t get_checked_int(json &object, std::string key) {
        check_existance(object, key);
        if (!object[key].is_number()) {
            std::cerr << "Field '" << key << "' has to be an integer, but is a "
                      << object[key].type_name() << "!" << std::endl;
            exit(1);
        }
        if (object[key].is_number_float()) {
            std::cerr << "Field '" << key << "' has to be an integer, but is a float!" << std::endl;
        }
        return object[key].get<index_t>();
    }

    static json &get_checked_object(json &object, std::string key) {
        check_existance(object, key);
        if (!object[key].is_object()) {
            std::cerr << "Field '" << key << "' has to be an object, but is a "
                      << object[key].type_name() << "!" << std::endl;
            exit(1);
        }
        return object[key];
    }

    static json &get_checked_array(json &object, std::string key) {
        check_existance(object, key);
        if (!object[key].is_array()) {
            std::cerr << "Field '" << key << "' has to be an array, but is a "
                      << object[key].type_name() << "!" << std::endl;
            exit(1);
        }
        return object[key];
    }

    void load_config(std::string path_to_config) {
        json config;
        try {
            if (path_to_config == "-") {
                config = json::parse(std::cin);
            } else {
                config = json::parse(std::ifstream(path_to_config));
            }
        } catch (nlohmann::detail::parse_error e) {
            std::cerr << "Illegal config file: " << e.what() << std::endl;
            exit(1);
        }

        tau = get_checked_float(config, "tau");
        dx = get_checked_float(config, "dx");

        json &time = get_checked_object(config, "time");
        t_cutoff_factor = get_checked_float(time, "t_cutoff");
        t_detect_factor = get_checked_float(time, "t_detect");
        t_max_factor = get_checked_float(time, "t_max");
        if (time.contains("t_snap")) {
            t_snap_factor = get_checked_float(time, "t_snap");
        }

        json &source = get_checked_object(config, "source");
        frequency = get_checked_float(source, "frequency");
        t_0_factor = get_checked_float(source, "phase");
        source_x = get_checked_float(source, "x");
        source_y = get_checked_float(source, "y");
        source_radius = get_checked_float(source, "radius");

        json rings_array = get_checked_array(config, "cavity_rings");
        if (rings_array.size() > max_n_rings) {
            std::cerr << "Illegal config file: Too many rings. This build only supports up to "
                      << max_n_rings << std::endl;
            exit(1);
        }

        rings.clear();
        rings.reserve(rings_array.size());
        for (auto it : rings_array) {
            rings.push_back(RingParameter(it));
        }
    }

    float t_cutoff_factor;

    float t_detect_factor;

    float t_max_factor;

    std::optional<float> t_snap_factor;

    float frequency;

    float t_0_factor;

    float source_x;

    float source_y;

    float source_radius;

    float dx;

    float tau;

    struct RingParameter {
        float width;
        RelMaterial material;

        RingParameter(json &object) : width(0.0), material(RelMaterial::perfect_metal()) {
            width = get_checked_float(object, "width");
            if (width < 0.0) {
                std::cerr << "Invalid config file: Cavity ring width may not be negative!"
                          << std::endl;
                exit(1);
            }

            float mu_r = get_checked_float(object, "mu_r");
            float eps_r = get_checked_float(object, "eps_r");
            float sigma = get_checked_float(object, "sigma");
            material = RelMaterial{mu_r, eps_r, sigma};
        }
    };

    std::vector<RingParameter> rings;

    std::string out_dir;

    float t_cutoff() const { return t_cutoff_factor * tau; }

    float t_detect() const { return t_detect_factor * tau; }

    float t_max() const { return t_max_factor * tau; }

    float t_0() const { return t_0_factor * tau; }

    uindex_t source_c() const { return uindex_t(float(grid_range()[0] / 2) + source_x / dx); }

    uindex_t source_r() const { return uindex_t(float(grid_range()[0] / 2) + source_y / dx); }

    float dt() const { return (dx / float(c0 * sqrt_2)) * 0.99; }

    uindex_t n_timesteps() const { return uindex_t(std::ceil(t_max() / dt())); }

    std::optional<uindex_t> n_snap_timesteps() const {
        if (t_snap_factor.has_value()) {
            return uindex_t(std::ceil((*t_snap_factor * tau) / dt()));
        } else {
            return std::nullopt;
        }
    }

    // Omega (?) in Hz.
    float omega() const { return 2.0 * pi * frequency; }

    cl::sycl::range<2> grid_range() const {
        float outer_radius = 0.0;
        for (auto ring : rings) {
            outer_radius += ring.width;
        }
        uindex_t width = uindex_t(std::ceil((2 * outer_radius / dx) + 2));
        uindex_t height = width;
        return cl::sycl::range<2>(width, height);
    }

    void print_configuration() const {
        std::cout << "Simulation Configuration:" << std::endl;
        std::cout << std::endl;

        std::cout << "# Timing" << std::endl;
        std::cout << "tau               = " << tau << " s" << std::endl;
        std::cout << "t_cutoff          = " << t_cutoff_factor << " tau = " << t_cutoff() << " s"
                  << std::endl;
        std::cout << "t_detect          = " << t_detect_factor << " tau = " << t_detect() << " s"
                  << std::endl;
        std::cout << "t_max             = " << t_max_factor << " tau = " << t_max() << " s"
                  << std::endl;
        std::cout << std::endl;

        std::cout << "# Source Wave" << std::endl;
        std::cout << "phase             = " << t_0_factor << " tau = " << t_0() << " s"
                  << std::endl;
        std::cout << "frequency         = " << frequency << " Hz" << std::endl;
        std::cout << std::endl;

        std::cout << "# Cavity" << std::endl;
        float inner_radius = 0.0;
        for (uindex_t i = 0; i < rings.size(); i++) {
            std::cout << "## Ring No. " << i << std::endl;
            std::cout << "distance range    = [" << inner_radius << ", "
                      << inner_radius + rings[i].width << "]" << std::endl;
            inner_radius += rings[i].width;
            std::cout << "mu_r              = " << rings[i].material.mu_r << std::endl;
            std::cout << "eps_r             = " << rings[i].material.eps_r << std::endl;
            std::cout << "sigma             = " << rings[i].material.sigma << std::endl;
            std::cout << std::endl;
        }

        std::cout << "# Execution parameters" << std::endl;
        std::cout << "dx                = " << dx << " m/cell" << std::endl;
        std::cout << "dt                = " << dt() << " s/iteration" << std::endl;
        std::cout << "grid w/h          = " << grid_range()[0] << " cells" << std::endl;
        std::cout << "n. timesteps      = " << n_timesteps() << std::endl;
        if (t_snap_factor.has_value()) {
            std::cout << "n. snap timesteps = " << n_snap_timesteps().value() << std::endl;
        }
        std::cout << std::endl;
    }
};