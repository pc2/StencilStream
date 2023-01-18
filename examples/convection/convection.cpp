#include <StencilStream/SimpleCPUExecutor.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>

using namespace stencil;

struct ThermalConvectionCell {

    ThermalConvectionCell()
        : T(0.0), Pt(0.0), Vx(0.0), Vy(0.0), Rx(0.0), Ry(0.0), tau_xx(0.0), tau_yy(0.0),
          sigma_xy(0.0), RogT(0.0), eta(0.0), delta_V(0.0), dVxd_tau(0.0), dVyd_tau(0.0), ErrV(0.0),
          ErrP(0.0), qTx(0.0), qTy(0.0), dT_dt(0.0) {}

    double T, Pt, Vx, Vy, Rx, Ry;
    double tau_xx, tau_yy, sigma_xy;
    double RogT, eta, delta_V;
    double dVxd_tau, dVyd_tau;
    double ErrV, ErrP;
    double qTx, qTy, dT_dt;
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

class PseudoTransientKernel {
  public:
    using Cell = ThermalConvectionCell;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr uindex_t n_subgenerations = 9;
    using TimeDependentValue = std::monostate;

    double nx, ny;
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

        if (stencil.subgeneration == 0) {
            // assign!(ErrV, Vy)
            if (c < nx && r < ny + 1) {
                new_cell.ErrV = ALL(Vy);
            }

        } else if (stencil.subgeneration == 1) {
            // assign!(ErrP, Pt)
            if (c < nx && r < ny) {
                new_cell.ErrP = ALL(Pt);
            }

        } else if (stencil.subgeneration == 2) {
            // compute_1!(...)
            if (c < nx && r < ny) {
                new_cell.RogT = roh0_g_alpha * ALL(T);
                new_cell.eta = eta0 * (1.0 - delta_eta_delta_T * (ALL(T) + deltaT / 2.0));
                new_cell.delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
                new_cell.Pt = ALL(Pt) + delta_tau_iter / beta * new_cell.delta_V;
                new_cell.tau_xx =
                    2.0 * new_cell.eta * (D_XA(Vx) / dx - (1.0 / 3.0) * new_cell.delta_V);
                new_cell.tau_yy =
                    2.0 * new_cell.eta * (D_YA(Vy) / dy - (1.0 / 3.0) * new_cell.delta_V);
            }
            if (c < nx - 1 && r < ny - 1) {
                new_cell.sigma_xy = 2.0 * new_cell.eta * (0.5 * (D_YI(Vx) / dy + D_XI(Vy) / dx));
            }

        } else if (stencil.subgeneration == 3) {
            // compute_2!(...)
            if (c < nx - 1 && r < ny - 2) {
                new_cell.Rx = 1.0 / rho * (D_XI(tau_xx) / dx + D_YA(sigma_xy) / dy - D_XI(Pt) / dx);
                new_cell.dVxd_tau = dampX * ALL(dVxd_tau) + new_cell.Rx * delta_tau_iter;
            }
            if (c < nx - 2 && r < ny - 1) {
                new_cell.Ry =
                    1.0 / rho *
                    (D_YI(tau_yy) / dy + D_XA(sigma_xy) / dx - D_YI(Pt) / dy + AV_YI(RogT));
                new_cell.dVyd_tau = dampY * ALL(dVyd_tau) + new_cell.Ry * delta_tau_iter;
            }

        } else if (stencil.subgeneration == 4) {
            // update_V!(...)
            // Index shift since the original instructions assigned all to inner.
            if (c >= 1 && r >= 1) {
                if (c < (nx + 1) - 1 && r < ny - 1) {
                    new_cell.Vx = ALL(Vx) + stencil[ID(-1, -1)].dVxd_tau * delta_tau_iter;
                }
                if (c < nx - 1 && r < (ny + 1) - 1) {
                    new_cell.Vy = ALL(Vy) + stencil[ID(-1, -1)].dVyd_tau * delta_tau_iter;
                }
            }

        } else if (stencil.subgeneration == 5) {
            // bc_y!(Vx)
            if (c < nx + 1 && r < ny) {
                if (r == 0) {
                    new_cell.Vx = stencil[ID(0, 1)].Vx;
                }
                if (r == ny - 1) {
                    new_cell.Vx = stencil[ID(0, -1)].Vx;
                }
            }

        } else if (stencil.subgeneration == 6) {
            // bc_x!(Vy)
            if (c < nx && r < ny + 1) {
                if (c == 0) {
                    new_cell.Vy = stencil[ID(1, 0)].Vy;
                }
                if (c == nx - 1) {
                    new_cell.Vy = stencil[ID(-1, 0)].Vy;
                }
            }

        } else if (stencil.subgeneration == 7) {
            // compute_error!(ErrV, Vy)
            if (c < nx && r < ny + 1) {
                new_cell.ErrV = ALL(ErrV) - ALL(Vy);
            }

        } else if (stencil.subgeneration == 8) {
            // compute_error!(ErrP, Pt)
            if (c < nx && r < ny) {
                new_cell.ErrP = ALL(ErrP) - ALL(Pt);
            }
        }

        return new_cell;
    }
};

int main() {
    // Physics - dimentionally independent scales
    double ly = 1.0;     // domain extend, m
    double eta0 = 1.0;   // viscosity, Pa*s
    double DcT = 1.0;    // heat diffusivity, m^2/s
    double deltaT = 1.0; // initial temperature perturbation K

    // Physics - nondim numbers
    double Ra = 1e7;  // Raleigh number = ρ0*g*α*ΔT*ly^3/η0/DcT
    double Pra = 1e3; // Prandtl number = η0/ρ0/DcT
    double ar = 3;    // aspect ratio

    // Physics - dimentionally dependent parameters
    double lx = ar * ly;  // domain extend, m
    double w = 1e-2 * ly; // initial perturbation standard deviation, m
    double roh0_g_alpha = Ra * eta0 * DcT / deltaT / std::pow(ly, 3); // thermal expansion
    double delta_eta_delta_T = 1e-10 / deltaT; // viscosity's temperature dependence

    // Numerics
    double nx = 96 * ar - 1;
    double ny = 96 - 1;     // numerical grid resolutions
    double iterMax = 50000; // maximal number of pseudo-transient iterations
    double nt = 3000;       // total number of timesteps
    double nout = 10;       // frequency of plotting
    double nerr = 5;        // frequency of error checking
    double epsilon = 1e-4;  // nonlinear absolute tolerence
    double dmp = 2;         // damping paramter
    double st = 5;          // quiver plotting spatial step

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

    PseudoTransientKernel kernel{
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
    };

    cl::sycl::buffer<ThermalConvectionCell, 2> grid = cl::sycl::range<2>(nx + 1, ny + 1);
    {
        auto ac = grid.template get_access<cl::sycl::access::mode::discard_write>();
        for (uint32_t x = 0; x < nx + 1; x++) {
            for (uint32_t y = 0; y < ny + 1; y++) {
                ThermalConvectionCell cell;
                if (y == 0) {
                    cell.T = deltaT / 2.0;
                } else if (y == ny - 1) {
                    cell.T = -deltaT / 2.0;
                } else if (x < nx && y < ny) {
                    cell.T = deltaT * std::exp(-std::pow((x * dx - 0.5 * lx) / w, 2) -
                                               std::pow((y * dy - 0.5 * ly) / w, 2));
                }
                ac[x][y] = cell;
            }
        }
    }

    SimpleCPUExecutor<PseudoTransientKernel, tdv::NoneSupplier> executor(ThermalConvectionCell(),
                                                                         kernel);
    executor.set_input(grid);

    std::cout << "i\tVx\tVy\tRx\tRy" << std::endl;

    for (uint32_t it = 0; it < 1; it++) {
        double errV = 2 * epsilon;
        double errP = 2 * epsilon;
        double max_ErrV, max_ErrP, max_Vx, max_Vy, max_Pt;
        executor.set_i_generation(0);

        while ((errV > epsilon || errP > epsilon) && executor.get_i_generation() < iterMax) {
            executor.run(nerr);
            executor.copy_output(grid);

            max_ErrV = max_ErrP = max_Vx = max_Vy = max_Pt =
                -std::numeric_limits<double>::infinity();
            {
                auto ac = grid.template get_access<cl::sycl::access::mode::read>();
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
                std::cout << executor.get_i_generation() << "\t";
                std::cout << ac[30][30].Vx << "\t";
                std::cout << ac[30][30].Vy << "\t";
                std::cout << ac[30][30].Rx << "\t";
                std::cout << ac[30][30].Ry << std::endl;
            }
            errV = max_ErrV / (1e-12 + max_Vy);
            errP = max_ErrP / (1e-12 + max_Pt);
        }

        double dt_adv = std::min(dx / max_Vx, dy / max_Vy) / 2.1;
        double dt = std::min(dt_diff, dt_adv);

        // update T kernel
    }
    return 0;
}