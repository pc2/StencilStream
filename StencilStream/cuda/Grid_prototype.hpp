#pragma once
#include <memory>
#include <sycl/sycl.hpp>

template <typename Cell> class Grid {
  public:
    static constexpr std::size_t dimensions = 2;

    Grid(std::size_t r, std::size_t c)
        : buffer(sycl::range<2>(r, c)), T_out(sycl::range<1>(r * c)), Pt_out(sycl::range<1>(r * c)),
          Vx_out(sycl::range<1>(r * c)), Vy_out(sycl::range<1>(r * c)),
          tau_xx_out(sycl::range<1>(r * c)), tau_yy_out(sycl::range<1>(r * c)),
          sigma_xy_out(sycl::range<1>(r * c)), dVxd_tau_out(sycl::range<1>(r * c)),
          dVyd_tau_out(sycl::range<1>(r * c)), ErrV_out(sycl::range<1>(r * c)),
          ErrP_out(sycl::range<1>(r * c)) {}

    Grid(sycl::range<2> range, sycl::range<1> cell_range)
        : buffer(range), T_out(cell_range), Pt_out(cell_range), Vx_out(cell_range),
          Vy_out(cell_range), tau_xx_out(cell_range), tau_yy_out(cell_range),
          sigma_xy_out(cell_range), dVxd_tau_out(cell_range), dVyd_tau_out(cell_range),
          ErrV_out(cell_range), ErrP_out(cell_range) {}

    Grid(Grid const &other_grid)
        : buffer(other_grid.buffer), T_out(other_grid.T_out), Pt_out(other_grid.Pt_out),
          Vx_out(other_grid.Vx_out), Vy_out(other_grid.Vy_out), tau_xx_out(other_grid.tau_xx_out),
          tau_yy_out(other_grid.tau_yy_out), sigma_xy_out(other_grid.sigma_xy_out),
          dVxd_tau_out(other_grid.dVxd_tau_out), dVyd_tau_out(other_grid.dVyd_tau_out),
          ErrV_out(other_grid.ErrV_out), ErrP_out(other_grid.ErrP_out) {}

    template <sycl::access::mode access_mode = sycl::access::mode::read_write>
    class GridAccessor : public sycl::host_accessor<Cell, Grid::dimensions, access_mode> {
      public:
        /**
         * \brief Create a new accessor to the given grid.
         */
        GridAccessor(Grid &grid)
            : sycl::host_accessor<Cell, Grid::dimensions, access_mode>(grid.buffer) {}
    };

    std::size_t get_grid_height() const { return buffer.get_range()[0]; }

    std::size_t get_grid_width() const { return buffer.get_range()[1]; }

    Grid make_similar() const { return Grid(buffer.get_range(), T_out.get_range()); }

    sycl::buffer<Cell, 2> &get_buffer() { return buffer; }

    sycl::buffer<double, 1> &get_T_out_buffer() { return T_out; }
    sycl::buffer<double, 1> &get_Pt_out_buffer() { return Pt_out; }
    sycl::buffer<double, 1> &get_Vx_out_buffer() { return Vx_out; }
    sycl::buffer<double, 1> &get_Vy_out_buffer() { return Vy_out; }
    sycl::buffer<double, 1> &get_tau_xx_out_buffer() { return tau_xx_out; }
    sycl::buffer<double, 1> &get_tau_yy_out_buffer() { return tau_yy_out; }
    sycl::buffer<double, 1> &get_sigma_xy_out_buffer() { return sigma_xy_out; }
    sycl::buffer<double, 1> &get_dVxd_tau_out_buffer() { return dVxd_tau_out; }
    sycl::buffer<double, 1> &get_dVyd_tau_out_buffer() { return dVyd_tau_out; }
    sycl::buffer<double, 1> &get_ErrV_out_buffer() { return ErrV_out; }
    sycl::buffer<double, 1> &get_ErrP_out_buffer() { return ErrP_out; }

  private:
    sycl::buffer<Cell, 2> buffer;

    sycl::buffer<double, 1> T_out;
    sycl::buffer<double, 1> Pt_out;
    sycl::buffer<double, 1> Vx_out;
    sycl::buffer<double, 1> Vy_out;
    sycl::buffer<double, 1> tau_xx_out;
    sycl::buffer<double, 1> tau_yy_out;
    sycl::buffer<double, 1> sigma_xy_out;
    sycl::buffer<double, 1> dVxd_tau_out;
    sycl::buffer<double, 1> dVyd_tau_out;
    sycl::buffer<double, 1> ErrV_out;
    sycl::buffer<double, 1> ErrP_out;
};