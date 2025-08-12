#include <StencilStream/cuda/StencilUpdate_baseline.hpp>

#include <benchmark/benchmark.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <sycl/ext/intel/fpga_extensions.hpp>

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

#define ALL(FIELD) (stencil[0][0].FIELD)
#define INN(FIELD) (stencil[1][1].FIELD)
#define D_XA(FIELD) (stencil[1][0].FIELD - stencil[0][0].FIELD)
#define D_YA(FIELD) (stencil[0][1].FIELD - stencil[0][0].FIELD)
#define D_XI(FIELD) (stencil[1][1].FIELD - stencil[0][1].FIELD)
#define D_YI(FIELD) (stencil[1][1].FIELD - stencil[1][0].FIELD)
#define AV(FIELD)                                                                                  \
    ((stencil[0][0].FIELD + stencil[1][0].FIELD + stencil[0][1].FIELD + stencil[1][1].FIELD) * 0.25)
#define AV_YI(FIELD) ((stencil[1][0].FIELD + stencil[1][1].FIELD) * 0.5)

class PseudoTransientKernel {
  public:
    using Cell = ThermalConvectionCell;
    static constexpr std::size_t stencil_radius = 1;
    static constexpr size_t n_subiterations = 3;

    size_t nx, ny;
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
        Cell new_cell = stencil[0][0];
        size_t x = stencil.id[0];
        size_t y = stencil.id[1];

        if (stencil.subiteration == 0) {
            // assign!(ErrV, Vy)
            if (x < nx && y < ny + 1) {
                new_cell.ErrV = ALL(Vy);
            }

            // assign!(ErrP, Pt)
            if (x < nx && y < ny) {
                new_cell.ErrP = ALL(Pt);
            }

            // compute_1!(...)
            if (x < nx && y < ny) {
                double delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
                double eta = eta0 * (1.0 - delta_eta_delta_T * (ALL(T) + deltaT / 2.0));

                new_cell.Pt = ALL(Pt) - delta_tau_iter / beta * delta_V;
                new_cell.tau_xx = 2.0 * eta * (D_XA(Vx) / dx - (1.0 / 3.0) * delta_V);
                // The original implementation uses @av(eta) here, which would actually mean that
                // this computation should be moved one subiteration back. However, using @all(eta)
                // did not make a noticeable difference, which is why I'm using new_cell.eta here.
                new_cell.tau_yy = 2.0 * eta * (D_YA(Vy) / dy - (1.0 / 3.0) * delta_V);

                if (x < nx - 1 && y < ny - 1) {
                    new_cell.sigma_xy = eta * (D_YI(Vx) / dy + D_XI(Vy) / dx);
                }
            }

        } else if (stencil.subiteration == 1) {
            // compute_2!(...) and update_V!(...)
            if (x >= 1 && y >= 1) {
                if (x < (nx + 1) - 1 && y < ny - 1) {
                    double Rx = 1.0 / rho *
                                ((stencil[0][0].tau_xx - stencil[-1][0].tau_xx) / dx +
                                 (stencil[-1][0].sigma_xy - stencil[-1][-1].sigma_xy) / dy -
                                 (stencil[0][0].Pt - stencil[-1][0].Pt) / dx);
                    new_cell.dVxd_tau = dampX * ALL(dVxd_tau) + Rx * delta_tau_iter;
                    new_cell.Vx = ALL(Vx) + new_cell.dVxd_tau * delta_tau_iter;
                }
                if (x < nx - 1 && y < (ny + 1) - 1) {
                    double Ry = 1.0 / rho *
                                ((stencil[0][0].tau_yy - stencil[0][-1].tau_yy) / dy +
                                 (stencil[0][-1].sigma_xy - stencil[-1][-1].sigma_xy) / dx -
                                 (stencil[0][0].Pt - stencil[0][-1].Pt) / dy +
                                 roh0_g_alpha * ((stencil[0][-1].T + stencil[0][0].T) * 0.5));
                    new_cell.dVyd_tau = dampY * ALL(dVyd_tau) + Ry * delta_tau_iter;
                    new_cell.Vy = ALL(Vy) + new_cell.dVyd_tau * delta_tau_iter;
                }
            }

        } else if (stencil.subiteration == 2) {
            // bc_y!(Vx)
            if (x < nx + 1 && y < ny) {
                if (y == 0) {
                    new_cell.Vx = stencil[0][1].Vx;
                }
                if (y == ny - 1) {
                    new_cell.Vx = stencil[0][-1].Vx;
                }
            }

            // bc_x!(Vy)
            if (x < nx && y < ny + 1) {
                if (x == 0) {
                    new_cell.Vy = stencil[1][0].Vy;
                }
                if (x == nx - 1) {
                    new_cell.Vy = stencil[-1][0].Vy;
                }
            }

            // compute_error!(ErrV, Vy)
            if (x < nx && y < ny + 1) {
                new_cell.ErrV = ALL(ErrV) - new_cell.Vy;
            }

            // compute_error!(ErrP, Pt)
            if (x < nx && y < ny) {
                new_cell.ErrP = ALL(ErrP) - ALL(Pt);
            }
        }

        return new_cell;
    }
};

class ThermalSolverKernel {
  public:
    using Cell = ThermalConvectionCell;
    static constexpr std::size_t stencil_radius = 1;
    static constexpr size_t n_subiterations = 2;

    size_t nx, ny;
    double dx, dy, dt;
    double DcT;

    Cell operator()(Stencil<Cell, 1> const &stencil) const {
        Cell new_cell = stencil[0][0];
        size_t x = stencil.id[0];
        size_t y = stencil.id[1];

        if (stencil.subiteration == 0) {
            if (x > 0 && y > 0 && x < nx - 1 && y < ny - 1) {
                // We only need qTx and qTy in this iteration, so I'm moving them here.
                double qTx_top_left = -DcT * (stencil[0][0].T - stencil[-1][0].T) / dx;
                double qTx_top = -DcT * (stencil[1][0].T - stencil[0][0].T) / dx;

                double qTy_top_left = -DcT * (stencil[0][0].T - stencil[0][-1].T) / dy;
                double qTy_left = -DcT * (stencil[0][1].T - stencil[0][0].T) / dy;

                // advect_T!(...)
                // The indices in advect_T are shifted by -1 since the computation of T only uses
                // dT_dt from the (-1, -1) cell.
                double dT_dt = -((qTx_top - qTx_top_left) / dx + (qTy_left - qTy_top_left) / dy);
                if (stencil[0][0].Vx > 0) {
                    dT_dt -= stencil[0][0].Vx * (stencil[0][0].T - stencil[-1][0].T) / dx;
                }
                if (stencil[1][0].Vx < 0) {
                    dT_dt -= stencil[1][0].Vx * (stencil[1][0].T - stencil[0][0].T) / dx;
                }
                if (stencil[0][0].Vy > 0) {
                    dT_dt -= stencil[0][0].Vy * (stencil[0][0].T - stencil[0][-1].T) / dy;
                }
                if (stencil[0][1].Vy < 0) {
                    dT_dt -= stencil[0][1].Vy * (stencil[0][1].T - stencil[0][0].T) / dy;
                }

                // compute_qT!(...)
                new_cell.T = ALL(T) + dT_dt * dt;
            }

        } else if (stencil.subiteration == 1) {
            // no_fluxY_T!(...)
            if (x == nx - 1 && y < ny) {
                new_cell.T = stencil[-1][0].T;
            }
            if (x == 0 && y < ny) {
                new_cell.T = stencil[1][0].T;
            }
        }

        return new_cell;
    }
};

using ThermalGrid = Grid<ThermalConvectionCell>;
using PseudoTransientUpdate = StencilUpdate<PseudoTransientKernel>;
using ThermalSolverUpdate = StencilUpdate<ThermalSolverKernel>;

static void BM_HotspotKernel(benchmark::State &state) {
    auto main_time_start = std::chrono::high_resolution_clock::now();

    std::filesystem::path experiment_file_path(
        "/scratch/hpc-lco-kenter/tstoehr/sycl-stencil/examples/convection/experiments/"
        "max-res-default.json");
    std::filesystem::path output_dir_path("/scratch/hpc-lco-kenter/tstoehr/sycl-stencil/build/out");

    json experiment;
    std::ifstream experiment_file(experiment_file_path);
    experiment = json::parse(experiment_file);

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
    size_t res = experiment.at("res");
    size_t nx = res * lx - 1;
    size_t ny = res * ly - 1;                  // numerical grid resolutions
    size_t iterMax = experiment.at("iterMax"); // maximal number of pseudo-transient iterations
    size_t nt = experiment.at("nt");           // total number of timesteps
    size_t nout = experiment.at("nout");       // frequency of plotting

    // TODO: Changed nerr to custom benchmark args
    size_t nerr = state.range(0); // frequency of error checking

    double epsilon = experiment.at("epsilon"); // nonlinear absolute tolerence
    double dmp = experiment.at("dmp");         // damping paramter

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

    sycl::device device(sycl::gpu_selector_v);

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
        .blocking = true,
    });

    ThermalGrid grid(nx + 1, ny + 1);
    {
        ThermalGrid::GridAccessor<sycl::access::mode::read_write> ac(grid);
        for (size_t x = 0; x < nx + 1; x++) {
            for (size_t y = 0; y < ny + 1; y++) {
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

    double thermal_solver_kernel_walltime = 0.0;
    double thermal_solver_walltime = 0.0;
    double thermal_solver_data_preperation_time_before = 0.0;
    double thermal_solver_data_preperation_time_after = 0.0;
    double error_check_time = 0.0;
    int total_iterations = 0;

    for (auto _ : state) {
        ThermalGrid local_grid = grid;
        total_iterations = 0;

        for (size_t it = 1; it <= 5; it++) {
            double errV = 2 * epsilon;
            double errP = 2 * epsilon;
            double max_ErrV, max_ErrP, max_Vx, max_Vy, max_Pt;
            size_t iter;

            auto transients_start = std::chrono::high_resolution_clock::now();
            for (iter = 0; iter < iterMax && (errV > epsilon || errP > epsilon); iter += nerr) {
                local_grid = pseudo_transient_update(local_grid);

                max_ErrV = max_ErrP = max_Vx = max_Vy = max_Pt =
                    -std::numeric_limits<double>::infinity();

                ThermalGrid::GridAccessor<sycl::access::mode::read> ac(local_grid);
                for (size_t x = 0; x < nx + 1; x++) {
                    for (size_t y = 0; y < ny + 1; y++) {
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
                errV = max_ErrV / (1e-12 + max_Vy);
                errP = max_ErrP / (1e-12 + max_Pt);
            }

            double dt_adv = std::min(dx / max_Vx, dy / max_Vy) / 2.1;
            double dt = std::min(dt_diff, dt_adv);

            ThermalSolverUpdate thermal_solver_update({
                .transition_function =
                    ThermalSolverKernel{
                        .nx = nx, .ny = ny, .dx = dx, .dy = dy, .dt = dt, .DcT = DcT},
                .halo_value = ThermalConvectionCell::halo_value(),
                .n_iterations = 1,
                .device = device,
                .blocking = true,
            });
            local_grid = thermal_solver_update(local_grid);

            total_iterations += iter;

            printf("it = %zu (iter = %zu), errV=%1.3e, errP=%1.3e \n", it, iter, errV, errP);
        }
    }

    int bytes_per_cell = 88.0;
    int operations_per_cell = 67.0;

    // Benchmark output
    double kernel_time = pseudo_transient_update.get_kernel_runtime();
    double avg_kernel_time = kernel_time / state.iterations();
    double wall_time = pseudo_transient_update.get_walltime();
    double avg_wall_time = wall_time / state.iterations();
    double avg_data_prep_time =
        pseudo_transient_update.get_data_preperation_time_before() / state.iterations();

    double flops = state.iterations() * 1.0 * nx * ny * operations_per_cell * total_iterations;
    double factor = avg_wall_time / avg_kernel_time; // >1 = Overhead
    int numbers_of_conversions = total_iterations / nerr;

    state.counters["Avg_Kernel_time"] = avg_kernel_time / numbers_of_conversions;
    state.counters["Avg_Wall_time"] = avg_wall_time / numbers_of_conversions;
    state.counters["Sim_time"] = benchmark::Counter(nerr, benchmark::Counter::kDefaults);
    state.counters["Flops_walltime"] = flops / wall_time;
    state.counters["Flops_kerneltime"] = flops / kernel_time;
    state.counters["Kerneltime to walltime"] = factor;
    state.counters["Data_prep_time"] = avg_data_prep_time / numbers_of_conversions;
    state.counters["total_iterations"] = total_iterations;
}

void CustomArgs(benchmark::internal::Benchmark *b) {
    int iter = 8;
    while (iter <= 20) {
        b->Args({iter});
        iter += 1;
    }
}

BENCHMARK(BM_HotspotKernel)->Apply(CustomArgs)->Unit(benchmark::kSecond);

BENCHMARK_MAIN();
