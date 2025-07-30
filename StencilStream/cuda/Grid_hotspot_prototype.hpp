#pragma once
#include <memory>
#include <sycl/sycl.hpp>

template <typename Cell> class Grid {
  public:
    static constexpr std::size_t dimensions = 2;

    Grid(std::size_t r, std::size_t c)
        : buffer(sycl::range<2>(r, c)), temp_out(sycl::range<1>(r * c)),
          power_out(sycl::range<1>(r * c)) {}

    Grid(sycl::range<2> range, sycl::range<1> cell_range)
        : buffer(range), temp_out(cell_range), power_out(cell_range) {}

    Grid(Grid const &other_grid)
        : buffer(other_grid.buffer), temp_out(other_grid.temp_out),
          power_out(other_grid.power_out) {}

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

    sycl::buffer<double, 1> &get_temp_out_buffer() { return temp_out; }
    sycl::buffer<double, 1> &get_power_out_buffer() { return power_out; }

  private:
    sycl::buffer<Cell, 2> buffer;

    sycl::buffer<float, 1> temp_out;
    sycl::buffer<float, 1> power_out;
};