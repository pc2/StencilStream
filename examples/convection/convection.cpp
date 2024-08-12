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
#if defined(STENCILSTREAM_BACKEND_CPU)
    #include <StencilStream/cpu/StencilUpdate.hpp>
#else
    #include <StencilStream/monotile/StencilUpdate.hpp>
#endif
#include <StencilStream/BaseTransitionFunction.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace stencil;
using json = nlohmann::json;

struct ThermalConvectionCell {
    double T, Pt, Vx, Vy;
    double tau_xx, tau_yy, sigma_xy;
    double dVxd_tau, dVyd_tau;
    double ErrV, ErrP;

    static ThermalConvectionCell halo_value() {
        return ThermalConvectionCell{
            .T = 0.0,
            .Pt = 0.0,
            .Vx = 0.0,
            .Vy = 0.0,
            .tau_xx = 0.0,
            .tau_yy = 0.0,
            .sigma_xy = 0.0,
            .dVxd_tau = 0.0,
            .dVyd_tau = 0.0,
            .ErrV = 0.0,
            .ErrP = 0.0,
        };
    }
};

#define ALL(FIELD) (stencil[ID(0, 0)].FIELD)
#define INN(FIELD) (stencil[ID(1, 1)].FIELD)
#define D_XA(FIELD) (stencil[ID(1, 0)].FIELD - stencil[ID(0, 0)].FIELD)
#define D_YA(FIELD) (stencil[ID(0, 1)].FIELD - stencil[ID(0, 0)].FIELD)
#define D_XI(FIELD) (stencil[ID(1, 1)].FIELD - stencil[ID(0, 1)].FIELD)
#define D_YI(FIELD) (stencil[ID(1, 1)].FIELD - stencil[ID(1, 0)].FIELD)
#define AV(FIELD)                                                                                  \
    ((stencil[ID(0, 0)].FIELD + stencil[ID(1, 0)].FIELD + stencil[ID(0, 1)].FIELD +                \
      stencil[ID(1, 1)].FIELD) *                                                                   \
     0.25)
#define AV_YI(FIELD) ((stencil[ID(1, 0)].FIELD + stencil[ID(1, 1)].FIELD) * 0.5)

class PseudoTransientKernel : public BaseTransitionFunction {
  public:
    using Cell = ThermalConvectionCell;

    static constexpr uindex_t n_subiterations = 3;

    uindex_t nx, ny;
    double roh0_g_alpha;
    double delta_eta_delta_T;
    double eta0;
    double deltaT;
    double dx, dy;
    double delta_tau_iter;
    double beta;
    double rho;
    double dampX, dampY;
    double DcT;

    Cell operator()(Stencil<Cell, 1> const &stencil) const {
        Cell new_cell = stencil[ID(0, 0)];
        uindex_t c = stencil.id.c;
        uindex_t r = stencil.id.r;

        if (stencil.subiteration == 0) {
            // assign!(ErrV, Vy)
            if (c < nx && r < ny + 1) {
                new_cell.ErrV = ALL(Vy);
            }

            // assign!(ErrP, Pt)
            if (c < nx && r < ny) {
                new_cell.ErrP = ALL(Pt);
            }

            // compute_1!(...)
            if (c < nx && r < ny) {
                double delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
                double eta = eta0 * (1.0 - delta_eta_delta_T * (ALL(T) + deltaT / 2.0));

                new_cell.Pt = ALL(Pt) - delta_tau_iter / beta * delta_V;
                new_cell.tau_xx = 2.0 * eta * (D_XA(Vx) / dx - (1.0 / 3.0) * delta_V);
                // The original implementation uses @av(eta) here, which would actually mean that
                // this computation should be moved one subiteration back. However, using @all(eta)
                // did not make a noticeable difference, which is why I'm using new_cell.eta here.
                new_cell.tau_yy = 2.0 * eta * (D_YA(Vy) / dy - (1.0 / 3.0) * delta_V);

                if (c < nx - 1 && r < ny - 1) {
                    new_cell.sigma_xy = eta * (D_YI(Vx) / dy + D_XI(Vy) / dx);
                }
            }

        } else if (stencil.subiteration == 1) {
            // compute_2!(...) and update_V!(...)
            if (c >= 1 && r >= 1) {
                if (c < (nx + 1) - 1 && r < ny - 1) {
                    double Rx = 1.0 / rho *
                                ((stencil[ID(0, 0)].tau_xx - stencil[ID(-1, 0)].tau_xx) / dx +
                                 (stencil[ID(-1, 0)].sigma_xy - stencil[ID(-1, -1)].sigma_xy) / dy -
                                 (stencil[ID(0, 0)].Pt - stencil[ID(-1, 0)].Pt) / dx);
                    new_cell.dVxd_tau = dampX * ALL(dVxd_tau) + Rx * delta_tau_iter;
                    new_cell.Vx = ALL(Vx) + new_cell.dVxd_tau * delta_tau_iter;
                }
                if (c < nx - 1 && r < (ny + 1) - 1) {
                    double Ry =
                        1.0 / rho *
                        ((stencil[ID(0, 0)].tau_yy - stencil[ID(0, -1)].tau_yy) / dy +
                         (stencil[ID(0, -1)].sigma_xy - stencil[ID(-1, -1)].sigma_xy) / dx -
                         (stencil[ID(0, 0)].Pt - stencil[ID(0, -1)].Pt) / dy +
                         roh0_g_alpha * ((stencil[ID(0, -1)].T + stencil[ID(0, 0)].T) * 0.5));
                    new_cell.dVyd_tau = dampY * ALL(dVyd_tau) + Ry * delta_tau_iter;
                    new_cell.Vy = ALL(Vy) + new_cell.dVyd_tau * delta_tau_iter;
                }
            }

        } else if (stencil.subiteration == 2) {
            // bc_y!(Vx)
            if (c < nx + 1 && r < ny) {
                if (r == 0) {
                    new_cell.Vx = stencil[ID(0, 1)].Vx;
                }
                if (r == ny - 1) {
                    new_cell.Vx = stencil[ID(0, -1)].Vx;
                }
            }

            // bc_x!(Vy)
            if (c < nx && r < ny + 1) {
                if (c == 0) {
                    new_cell.Vy = stencil[ID(1, 0)].Vy;
                }
                if (c == nx - 1) {
                    new_cell.Vy = stencil[ID(-1, 0)].Vy;
                }
            }

            // compute_error!(ErrV, Vy)
            if (c < nx && r < ny + 1) {
                new_cell.ErrV = ALL(ErrV) - new_cell.Vy;
            }

            // compute_error!(ErrP, Pt)
            if (c < nx && r < ny) {
                new_cell.ErrP = ALL(ErrP) - ALL(Pt);
            }
        }

        return new_cell;
    }
};

class ThermalSolverKernel : public BaseTransitionFunction {
  public:
    using Cell = ThermalConvectionCell;

    static constexpr uindex_t n_subiterations = 2;

    uindex_t nx, ny;
    double dx, dy, dt;
    double DcT;

    Cell operator()(Stencil<Cell, 1> const &stencil) const {
        Cell new_cell = stencil[ID(0, 0)];
        uindex_t c = stencil.id.c;
        uindex_t r = stencil.id.r;

        if (stencil.subiteration == 0) {
            if (c > 0 && r > 0 && c < nx - 1 && r < ny - 1) {
                // We only need qTx and qTy in this iteration, so I'm moving them here.
                double qTx_top_left = -DcT * (stencil[ID(0, 0)].T - stencil[ID(-1, 0)].T) / dx;
                double qTx_top = -DcT * (stencil[ID(1, 0)].T - stencil[ID(0, 0)].T) / dx;

                double qTy_top_left = -DcT * (stencil[ID(0, 0)].T - stencil[ID(0, -1)].T) / dy;
                double qTy_left = -DcT * (stencil[ID(0, 1)].T - stencil[ID(0, 0)].T) / dy;

                // advect_T!(...)
                // The indices in advect_T are shifted by -1 since the computation of T only uses
                // dT_dt from the (-1, -1) cell.
                double dT_dt = -((qTx_top - qTx_top_left) / dx + (qTy_left - qTy_top_left) / dy);
                if (stencil[ID(0, 0)].Vx > 0) {
                    dT_dt -=
                        stencil[ID(0, 0)].Vx * (stencil[ID(0, 0)].T - stencil[ID(-1, 0)].T) / dx;
                }
                if (stencil[ID(1, 0)].Vx < 0) {
                    dT_dt -=
                        stencil[ID(1, 0)].Vx * (stencil[ID(1, 0)].T - stencil[ID(0, 0)].T) / dx;
                }
                if (stencil[ID(0, 0)].Vy > 0) {
                    dT_dt -=
                        stencil[ID(0, 0)].Vy * (stencil[ID(0, 0)].T - stencil[ID(0, -1)].T) / dy;
                }
                if (stencil[ID(0, 1)].Vy < 0) {
                    dT_dt -=
                        stencil[ID(0, 1)].Vy * (stencil[ID(0, 1)].T - stencil[ID(0, 0)].T) / dy;
                }

                // compute_qT!(...)
                new_cell.T = ALL(T) + dT_dt * dt;
            }

        } else if (stencil.subiteration == 1) {
            // no_fluxY_T!(...)
            if (c == nx - 1 && r < ny) {
                new_cell.T = stencil[ID(-1, 0)].T;
            }
            if (c == 0 && r < ny) {
                new_cell.T = stencil[ID(1, 0)].T;
            }
        }

        return new_cell;
    }
};

#if defined(STENCILSTREAM_BACKEND_CPU)
using Grid = cpu::Grid<ThermalConvectionCell>;
using PseudoTransientUpdate = cpu::StencilUpdate<PseudoTransientKernel>;
using ThermalSolverUpdate = cpu::StencilUpdate<ThermalSolverKernel>;

#else
constexpr uindex_t max_grid_width = 1 << 16;
constexpr uindex_t max_grid_height = 512;
using Grid = monotile::Grid<ThermalConvectionCell>;
using PseudoTransientUpdate =
    monotile::StencilUpdate<PseudoTransientKernel, PseudoTransientKernel::n_subiterations * 8,
                            max_grid_width, max_grid_height>;
using ThermalSolverUpdate =
    monotile::StencilUpdate<ThermalSolverKernel, ThermalSolverKernel::n_subiterations,
                            max_grid_width, max_grid_height>;

#endif

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path to experiment>.json <path to output directory>"
                  << std::endl;
        return 1;
    }

    std::filesystem::path experiment_file_path(argv[1]);
    std::filesystem::path output_dir_path(argv[2]);

    if (!std::filesystem::is_regular_file(experiment_file_path)) {
        std::cerr << "The experiment file does not exist or is not a regular file." << std::endl;
        return 1;
    }

    if (!std::filesystem::is_directory(output_dir_path)) {
        std::cerr << "The output directory does not exist or is not a directory." << std::endl;
        return 1;
    }

    std::ifstream experiment_file(experiment_file_path);
    if (!experiment_file.is_open()) {
        std::cerr << "Could not open experiment file!" << std::endl;
        return 1;
    }

    json experiment;
    try {
        experiment = json::parse(experiment_file);
    } catch (json::parse_error e) {
        std::cerr << "Could not parse experiment file:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Physics - dimentionally independent scales
    double lx = experiment.at("lx");         // horizontal domain extend, m
    double ly = experiment.at("ly");         // vertical domain extend, m
    double px = experiment.at("px");         // horizontal position of starting blob, m
    double py = experiment.at("py");         // vertical position of starting blob, m
    double eta0 = experiment.at("eta0");     // viscosity, Pa*s
    double DcT = experiment.at("DcT");       // heat diffusivity, m^2/s
    double deltaT = experiment.at("deltaT"); // initial temperature perturbation K

    // Physics - nondim numbers
    double Ra = experiment.at("Ra");   // Raleigh number = ρ0*g*α*ΔT*ly^3/η0/DcT
    double Pra = experiment.at("Pra"); // Prandtl number = η0/ρ0/DcT
    double ar = lx / ly;               // aspect ratio

    // Physics - dimentionally dependent parameters
    double w = 1e-2 * ly; // initial perturbation standard deviation, m
    double roh0_g_alpha = Ra * eta0 * DcT / deltaT / std::pow(ly, 3); // thermal expansion
    double delta_eta_delta_T = 1e-10 / deltaT; // viscosity's temperature dependence

    // Numerics
    uindex_t res = experiment.at("res");
    uindex_t nx = res * lx - 1;
    uindex_t ny = res * ly - 1;                  // numerical grid resolutions
    uindex_t iterMax = experiment.at("iterMax"); // maximal number of pseudo-transient iterations
    uindex_t nt = experiment.at("nt");           // total number of timesteps
    uindex_t nout = experiment.at("nout");       // frequency of plotting
    uindex_t nerr = experiment.at("nerr");       // frequency of error checking
    double epsilon = experiment.at("epsilon");   // nonlinear absolute tolerence
    double dmp = experiment.at("dmp");           // damping paramter

    // Derived numerics
    double dx = lx / (nx - 1);
    double dy = ly / (ny - 1);           // cell size
    double rho = 1.0 / Pra * eta0 / DcT; // density
    double dt_diff =
        1.0 / 4.1 * std::pow(std::min(dx, dy), 2) / DcT; // diffusive CFL timestep limiter
    double delta_tau_iter = 1.0 / 6.1 * std::min(dx, dy) /
                            std::sqrt(eta0 / rho); // iterative CFL pseudo-timestep limiter
    double beta = 6.1 * std::pow(delta_tau_iter, 2) / std::pow(std::min(dx, dy), 2) /
                  rho;             // numerical bulk compressibility
    double dampX = 1.0 - dmp / nx; // damping term for the x-momentum equation
    double dampY = 1.0 - dmp / ny; // damping term for the y-momentum equation

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    PseudoTransientUpdate pseudo_transient_update({
        .transition_function =
            PseudoTransientKernel{
                .nx = nx,
                .ny = ny,
                .roh0_g_alpha = roh0_g_alpha,
                .delta_eta_delta_T = delta_eta_delta_T,
                .eta0 = eta0,
                .deltaT = deltaT,
                .dx = dx,
                .dy = dy,
                .delta_tau_iter = delta_tau_iter,
                .beta = beta,
                .rho = rho,
                .dampX = dampX,
                .dampY = dampY,
                .DcT = DcT,
            },
        .halo_value = ThermalConvectionCell::halo_value(),
        .n_iterations = nerr,
        .device = device,
    });

    Grid grid(nx + 1, ny + 1);
    {
        Grid::GridAccessor<sycl::access::mode::read_write> ac(grid);
        for (uint32_t x = 0; x < nx + 1; x++) {
            for (uint32_t y = 0; y < ny + 1; y++) {
                ThermalConvectionCell cell = ThermalConvectionCell::halo_value();
                if (y == 0) {
                    cell.T = deltaT / 2.0;
                } else if (y == ny - 1) {
                    cell.T = -deltaT / 2.0;
                } else if (x < nx && y < ny) {
                    cell.T = deltaT * std::exp(-std::pow((x * dx - px) / w, 2) -
                                               std::pow((y * dy - py) / w, 2));
                }
                ac[x][y] = cell;
            }
        }
    }

    // Starting iteration with one and using an inclusive upper bound to stay compatible with the
    // reference.
    auto computation_start = std::chrono::system_clock::now();
    for (uint32_t it = 1; it <= nt; it++) {
        double errV = 2 * epsilon;
        double errP = 2 * epsilon;
        double max_ErrV, max_ErrP, max_Vx, max_Vy, max_Pt;
        uindex_t iter;

        auto transients_start = std::chrono::high_resolution_clock::now();
        for (iter = 0; iter < iterMax && (errV > epsilon || errP > epsilon); iter += nerr) {
            grid = pseudo_transient_update(grid);

            max_ErrV = max_ErrP = max_Vx = max_Vy = max_Pt =
                -std::numeric_limits<double>::infinity();
            {
                Grid::GridAccessor<sycl::access::mode::read> ac(grid);
                for (uint32_t x = 0; x < nx + 1; x++) {
                    for (uint32_t y = 0; y < ny + 1; y++) {
                        auto cell = ac[x][y];
                        if (x < nx && y < ny + 1 && std::abs(cell.ErrV) > max_ErrV) {
                            max_ErrV = std::abs(cell.ErrV);
                        }
                        if (x < nx && y < ny && std::abs(cell.ErrP) > max_ErrP) {
                            max_ErrP = std::abs(cell.ErrP);
                        }
                        if (x < nx + 1 && y < ny && std::abs(cell.Vx) > max_Vx) {
                            max_Vx = std::abs(cell.Vx);
                        }
                        if (x < nx && y < ny && std::abs(cell.Vy) > max_Vy) {
                            max_Vy = std::abs(cell.Vy);
                        }
                        if (x < nx && y < ny && std::abs(cell.Pt) > max_Pt) {
                            max_Pt = std::abs(cell.Pt);
                        }
                    }
                }
            }
            errV = max_ErrV / (1e-12 + max_Vy);
            errP = max_ErrP / (1e-12 + max_Pt);
        }
        auto transients_end = std::chrono::high_resolution_clock::now();

        auto transients_computation_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(transients_end -
                                                                      transients_start);

        printf("it = %d (iter = %d, time = %e), errV=%1.3e, errP=%1.3e \n", it, iter,
               transients_computation_time.count(), errV, errP);

        double dt_adv = std::min(dx / max_Vx, dy / max_Vy) / 2.1;
        double dt = std::min(dt_diff, dt_adv);

        ThermalSolverUpdate thermal_solver_update({
            .transition_function =
                ThermalSolverKernel{.nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .DcT = DcT},
            .halo_value = ThermalConvectionCell::halo_value(),
            .n_iterations = 1,
            .device = device,
        });
        grid = thermal_solver_update(grid);

        if (it > 0 && it % nout == 0) {
            std::filesystem::path output_file_path =
                output_dir_path / std::filesystem::path(std::to_string(it) + ".csv");
            std::ofstream out_file(output_file_path);
            {
                Grid::GridAccessor<sycl::access::mode::read> ac(grid);
                for (uindex_t c = 0; c < nx; c++) {
                    for (uindex_t r = 0; r < ny; r++) {
                        out_file << ac[c][r].T;
                        if (r != ny - 1) {
                            out_file << ",";
                        }
                    }
                    out_file << "\n";
                }
            }
            out_file.close();
        }
    }

    auto computation_end = std::chrono::system_clock::now();
    auto computation_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        computation_end - computation_start);
    std::cout << "Total time = " << computation_time.count() << std::endl;
    return 0;
}
