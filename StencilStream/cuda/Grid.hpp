#pragma once
#include <memory>
#include <sycl/sycl.hpp>

template <typename Cell> class Grid {
  public:
    static constexpr std::size_t dimensions = 2;

    Grid(std::size_t r, std::size_t c) : buffer(sycl::range<2>(r, c)) {}

    Grid(sycl::range<2> range) : buffer(range) {}

    Grid(sycl::buffer<Cell, 2> other_buffer) : buffer(other_buffer.get_range()) {
        copy_from_buffer(other_buffer);
    }

    Grid(Grid const &other_grid) : buffer(other_grid.buffer) {}

    void copy_from_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::write_only);
        sycl::host_accessor other_ac(other_buffer, sycl::read_only);
        std::memcpy(buffer_ac.get_pointer(), other_ac.get_pointer(), buffer_ac.byte_size());
    }

    void copy_to_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::read_only);
        sycl::host_accessor other_ac(other_buffer, sycl::write_only);
        std::memcpy(other_ac.get_pointer(), buffer_ac.get_pointer(), buffer_ac.byte_size());
    }

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

    Grid make_similar() const { return Grid(buffer.get_range()); }

    sycl::buffer<Cell, 2> &get_buffer() { return buffer; }

  private:
    sycl::buffer<Cell, 2> buffer;
};