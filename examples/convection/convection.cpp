#include <StencilStream/SimpleCPUExecutor.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>

using namespace stencil;

struct ThermalConvectionCell {
    float T, Pt, Vx, Vy, Rx, Ry;
    float tau_xx, tau_yy, sigma_xy;
    float eta;
    float dVxd_tau, dVyd_tau;
    float ErrV, ErrP;
    float qTx, qTy, dT_dt;
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
    float DcT;

    Cell operator()(Stencil<Cell, 2> const &stencil) const {
        Cell new_cell = stencil[ID(0, 0)];
        if (stencil.subgeneration == 0) {
            // assign!(ErrV, Vy)
            new_cell.ErrV = stencil[ID(0, 0)].Vy;
            // assign!(ErrP, Pt)
            new_cell.ErrP = stencil[ID(0, 0)].Pt;

            // compute_1!(...) (part 1 of 2)
            new_cell.eta = eta0 * (1.0 - delta_eta_delta_T * (stencil[ID(0, 0)].T + deltaT / 2.0));
            float delta_V = D_XA(Vx) / dx + D_YA(Vy) / dy;
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
    SimpleCPUExecutor<PseudoTransientKernel, tdv::NoneSupplier> executor(ThermalConvectionCell{},
                                                                         PseudoTransientKernel());
    return 0;
}