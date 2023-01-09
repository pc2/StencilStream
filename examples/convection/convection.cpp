#include <StencilStream/MonotileExecutor.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>

using namespace stencil;

struct ThermalConvectionCell {
    float T, Pt, Vx, Vy, Rx, Ry;
    float tau_xx, tau_yy, sigma_xy;
    float eta;
    float dVxd_tau, dVyd_tau;
    float ErrV, ErrP;
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

    float roh0_g_alpha;
    float delta_eta_delta_T;
    float eta0;
    float deltaT;
    float dx, dy;
    float Pt;
    float delta_tau_iter;
    float beta;
    float rho;
    float dampX, dampY;

    Cell operator()(Stencil<Cell, 2> const &stencil) const {
        Cell new_cell = stencil[ID(0, 0)];
        if (stencil.subgeneration == 0) {
            new_cell.ErrV = stencil[ID(0, 0)].Vy;
            new_cell.ErrP = stencil[ID(0, 0)].Pt;

            new_cell.eta = eta0 * (1.0 - delta_eta_delta_T * (stencil[ID(0, 0)].T + deltaT / 2.0));
            float delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
            new_cell.Pt -= delta_tau_iter / beta * delta_V;
            new_cell.tau_xx = 2.0 * new_cell.eta * (D_XA(Vx) / dx - 1.0 / 3.0 * delta_V);
            new_cell.tau_yy = 2.0 * new_cell.eta * (D_YA(Vy) / dy - 1.0 / 3.0 * delta_V);

            new_cell.ErrP -= new_cell.Pt;
        } else if (stencil.subgeneration == 1) {
            new_cell.sigma_xy = 2.0 * AV(eta) * (0.5 * (D_YI(Vx) / dy + D_XI(Vy) / dx));
        } else if (stencil.subgeneration == 2) {
            new_cell.Rx = 1.0 / rho * (D_XI(tau_xx) / dx + D_YA(sigma_xy) / dy - D_XI(Pt) / dx);
            new_cell.Ry =
                1.0 / rho *
                (D_YI(tau_yy) / dy + D_XA(sigma_xy) / dx - D_YI(Pt) / dy + roh0_g_alpha * AV_YI(T));
            new_cell.dVxd_tau = dampX * stencil[ID(0, 0)].dVxd_tau + new_cell.Rx * delta_tau_iter;
            new_cell.dVyd_tau = dampY * stencil[ID(0, 0)].dVyd_tau + new_cell.Ry * delta_tau_iter;
        } else if (stencil.subgeneration == 3) {
            // Originally, this is an assignment to @inn(Vx) and @inn(Vy), I may have messed up the
            // indices here.
            new_cell.Vx += stencil[ID(-1, -1)].Rx * delta_eta_delta_T;
            new_cell.Vy += stencil[ID(-1, -1)].Ry * delta_tau_iter;
        } else if (stencil.subgeneration == 4) {
            if (stencil.id.r == 0) {
                new_cell.Vx = stencil[ID(0, 1)].Vx;
            }
            if (stencil.id.r == stencil.grid_range.r - 1) {
                new_cell.Vx = stencil[ID(0, -1)].Vx;
            }
            if (stencil.id.c == 0) {
                new_cell.Vy = stencil[ID(1, 0)].Vy;
            }
            if (stencil.id.c == stencil.grid_range.c - 1) {
                new_cell.Vy = stencil[ID(-1, 0)].Vy;
            }
            new_cell.ErrV -= new_cell.Vy;
        }
        return new_cell;
    }
};

int main() {
    MonotileExecutor<PseudoTransientKernel, tdv::NoneSupplier, 5> executor(
        ThermalConvectionCell{}, PseudoTransientKernel());
    return 0;
}