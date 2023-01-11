#include <StencilStream/SimpleCPUExecutor.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>

using namespace stencil;

struct ThermalConvectionCell {

    ThermalConvectionCell()
        : T(0.0), Pt(0.0), Vx(0.0), Vy(0.0), Rx(0.0), Ry(0.0), tau_xx(0.0), tau_yy(0.0),
          sigma_xy(0.0), eta(0.0), dVxd_tau(0.0), dVyd_tau(0.0), ErrV(0.0), ErrP(0.0), qTx(0.0),
          qTy(0.0), dT_dt(0.0) {}

    double T, Pt, Vx, Vy, Rx, Ry;
    double tau_xx, tau_yy, sigma_xy;
    double eta;
    double dVxd_tau, dVyd_tau;
    double ErrV, ErrP;
    double qTx, qTy, dT_dt;
};

#define D_XA(FIELD) (stencil[ID(1, 0)].FIELD - stencil[ID(0, 0)].FIELD)
#define D_YA(FIELD) (stencil[ID(0, 1)].FIELD - stencil[ID(0, 0)].FIELD)
#define D_XI(FIELD) (stencil[ID(2, 1)].FIELD - stencil[ID(1, 1)].FIELD)
#define D_YI(FIELD) (stencil[ID(1, 2)].FIELD - stencil[ID(1, 1)].FIELD)
#define AV(FIELD)                                                                                  \
    (0.25 * (stencil[ID(0, 0)].FIELD + stencil[ID(1, 0)].FIELD + stencil[ID(0, 1)].FIELD +         \
             stencil[ID(1, 1)].FIELD))
#define AV_YI(FIELD) (0.5 * (stencil[ID(1, 0)].FIELD + stencil[ID(1, 1)].FIELD))

class PseudoTransientKernel {
  public:
    using Cell = ThermalConvectionCell;

    static constexpr uindex_t stencil_radius = 2;
    static constexpr uindex_t n_subgenerations = 5;
    using TimeDependentValue = std::monostate;

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

    Cell operator()(Stencil<Cell, 2> const &stencil) const {
        Cell new_cell = stencil[ID(0, 0)];
        if (stencil.subgeneration == 0) {
            // assign!(ErrV, Vy)
            new_cell.ErrV = stencil[ID(0, 0)].Vy;
            // assign!(ErrP, Pt)
            new_cell.ErrP = stencil[ID(0, 0)].Pt;

            // compute_1!(...) (part 1 of 2)
            new_cell.eta = eta0 * (1.0 - delta_eta_delta_T * (stencil[ID(0, 0)].T + deltaT / 2.0));
            double delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
            new_cell.Pt -= delta_tau_iter / beta * delta_V;
            new_cell.tau_xx = 2.0 * new_cell.eta * (D_XA(Vx) / dx - 1.0 / 3.0 * delta_V);
            new_cell.tau_yy = 2.0 * new_cell.eta * (D_YA(Vy) / dy - 1.0 / 3.0 * delta_V);

            // compute_error!(ErrP, Pt)
            new_cell.ErrP -= new_cell.Pt;

            // compute_qT!(...)
            new_cell.qTx = -DcT * D_XI(T) / dy;
            new_cell.qTy = -DcT * D_YI(T) / dy;
        } else if (stencil.subgeneration == 1) {
            // compute_1!(...) (part 2 of 2)
            new_cell.sigma_xy = 2.0 * AV(eta) * (0.5 * (D_YI(Vx) / dy + D_XI(Vy) / dx));

            // advect_T!(...)
            new_cell.dT_dt = -(D_XA(qTx) / dx + D_YA(qTy) / dy);
            // (Vx[ix+1,iy+1]>0)*Vx[ix+1,iy+1]*(T[ix+1,iy+1]-T[ix  ,iy+1])/dx
            if (stencil[ID(1, 1)].Vx > 0.0) {
                new_cell.dT_dt +=
                    stencil[ID(1, 1)].Vx * (stencil[ID(1, 1)].T - stencil[ID(0, 1)].T) / dx;
            }
            // (Vx[ix+2,iy+1]<0)*Vx[ix+2,iy+1]*(T[ix+2,iy+1]-T[ix+1,iy+1])/dx
            if (stencil[ID(2, 1)].Vx < 0.0) {
                new_cell.dT_dt +=
                    stencil[ID(2, 1)].Vx * (stencil[ID(2, 1)].T - stencil[ID(1, 1)].T) / dx;
            }
            // (Vy[ix+1,iy+1]>0)*Vy[ix+1,iy+1]*(T[ix+1,iy+1]-T[ix+1,iy  ])/dy
            if (stencil[ID(1, 1)].Vy > 0.0) {
                new_cell.dT_dt +=
                    stencil[ID(1, 1)].Vy * (stencil[ID(1, 1)].T - stencil[ID(1, 0)].T) / dy;
            }
            // (Vy[ix+1,iy+2]<0)*Vy[ix+1,iy+2]*(T[ix+1,iy+2]-T[ix+1,iy+1])/dy
            if (stencil[ID(1, 2)].Vy < 0.0) {
                new_cell.dT_dt +=
                    stencil[ID(1, 2)].Vy * (stencil[ID(1, 2)].T - stencil[ID(1, 1)].T) / dy;
            }
        } else if (stencil.subgeneration == 2) {
            // compute_2!(...)
            new_cell.Rx = 1.0 / rho * (D_XI(tau_xx) / dx + D_YA(sigma_xy) / dy - D_XI(Pt) / dx);
            new_cell.Ry =
                1.0 / rho *
                (D_YI(tau_yy) / dy + D_XA(sigma_xy) / dx - D_YI(Pt) / dy + roh0_g_alpha * AV_YI(T));
            new_cell.dVxd_tau = dampX * stencil[ID(0, 0)].dVxd_tau + new_cell.Rx * delta_tau_iter;
            new_cell.dVyd_tau = dampY * stencil[ID(0, 0)].dVyd_tau + new_cell.Ry * delta_tau_iter;
        } else if (stencil.subgeneration == 3) {
            // update_V!(...)
            new_cell.Vx += stencil[ID(-1, -1)].Rx * delta_eta_delta_T;
            new_cell.Vy += stencil[ID(-1, -1)].Ry * delta_tau_iter;
        } else if (stencil.subgeneration == 4) {
            // bc_y!(Vx)
            if (stencil.id.r == 0) {
                new_cell.Vx = stencil[ID(0, 1)].Vx;
            }
            if (stencil.id.r == stencil.grid_range.r - 1) {
                new_cell.Vx = stencil[ID(0, -1)].Vx;
            }

            // bc_x!(Vy)
            if (stencil.id.c == 0) {
                new_cell.Vy = stencil[ID(1, 0)].Vy;
            }
            if (stencil.id.c == stencil.grid_range.c - 1) {
                new_cell.Vy = stencil[ID(-1, 0)].Vy;
            }

            // compute_error!(ErrV, Vy)
            new_cell.ErrV -= new_cell.Vy;
        }
        return new_cell;
    }
};

int main() {
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
    double roh0_g_alpha = Ra * eta0 * DcT / deltaT / (ly * ly * ly); // thermal expansion
    double delta_eta_delta_T = 1e-10 / deltaT; // viscosity's temperature dependence

    // Numerics
    double nx = 96 * ar - 1;
    double ny = 96 - 1;     // numerical grid resolutions
    double iterMax = 50000; // maximal number of pseudo-transient iterations
    double nt = 3000;       // total number of timesteps
    double nout = 10;       // frequency of plotting
    double nerr = 100;      // frequency of error checking
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
                cell.T = deltaT * std::exp(-std::pow((x * dx - 0.5 * lx) / w, 2) -
                                           std::pow((y * dy - 0.5 * ly) / w, 2));
                ac[x][y] = cell;
            }
        }
    }

    SimpleCPUExecutor<PseudoTransientKernel, tdv::NoneSupplier> executor(ThermalConvectionCell(),
                                                                         kernel);
    executor.set_input(grid);

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
                for (uint32_t x = 0; x < nx; x++) {
                    for (uint32_t y = 0; y < ny; y++) {
                        auto cell = ac[x][y];
                        if (std::abs(cell.ErrV) > max_ErrV) {
                            max_ErrV = std::abs(cell.ErrV);
                        }
                        if (std::abs(cell.ErrP) > max_ErrP) {
                            max_ErrP = std::abs(cell.ErrP);
                        }
                        if (std::abs(cell.Vx) > max_Vx) {
                            max_Vx = std::abs(cell.Vx);
                        }
                        if (std::abs(cell.Vy) > max_Vy) {
                            max_Vy = std::abs(cell.Vy);
                        }
                        if (std::abs(cell.Pt) > max_Pt) {
                            max_Pt = std::abs(cell.Pt);
                        }
                    }
                }
            }
            errV = max_ErrV / (1e-12 + max_Vy);
            errP = max_ErrP / (1e-12 + max_Pt);
        }

        double dt_adv = std::min(dx / max_Vx, dy / max_Vy) / 2.1;
        double dt = std::min(dt_diff, dt_adv);

        std::cout << executor.get_i_generation() << " " << max_ErrV << " " << max_ErrP << std::endl;
        // update T kernel
    }
    return 0;
}