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
#include "../AccessorSubscript.hpp"
#include "../Concepts.hpp"
#include "../GenericID.hpp"
#include <memory>
#include <numeric>
#include <vector>

namespace stencil {
namespace tiling {

/**
 * \brief A rectangular container of cells with a dynamic, arbitrary size, used by the \ref
 * StencilExecutor.
 *
 * This class is part of the \ref tiling architecture. It logically contains the grid the transition
 * function is applied to and it partitions the grid into tiles of static size. These are the units
 * the \ref ExecutionKernel works on.
 *
 * Apart from providing copy operations to and from monolithic grid buffers, it also handles the
 * input and output kernel submission for a given tile.
 *
 * \tparam Cell Cell value type.
 * \tparam tile_width The number of columns of a tile.
 * \tparam tile_height The number of rows of a tile.
 * \tparam halo_radius The radius (aka width and height) of the tile halo.
 */
template <typename Cell, uindex_t tile_width = 1024, uindex_t tile_height = 1024,
          uindex_t halo_radius = 1>
class Grid {
  public:
    static_assert(2 * halo_radius < tile_height && 2 * halo_radius < tile_width);

    /**
     * \brief Create a grid with undefined contents.
     *
     * This constructor is used to create the output grid of a \ref ExecutionKernel
     * invocation. It's contents do not need to be initialized or copied from another buffer since
     * it will override cell values from the execution kernel anyway.
     *
     * \param grid_width The number of columns of the grid.
     * \param grid_height The number of rows of the grid.
     */
    Grid(uindex_t grid_width, uindex_t grid_height)
        : grid_buffer(sycl::range<2>(grid_width + 2 * halo_radius, grid_height + 2 * halo_radius)),
          grid_width(grid_width), grid_height(grid_height) {}

    Grid(sycl::range<2> range) : Grid(range[0], range[1]) {}

    Grid(sycl::buffer<Cell, 2> input_buffer) : Grid(input_buffer.get_range()) {
        copy_from_buffer(input_buffer);
    }

    Grid(Grid const &other_grid)
        : grid_buffer(other_grid.grid_buffer), grid_width(other_grid.grid_width),
          grid_height(other_grid.grid_height) {}

    template <sycl::access::mode access_mode> class GridAccessor {
      public:
        static constexpr uindex_t dimensions = 2;

        GridAccessor(Grid &grid)
            : accessor(grid.grid_buffer), grid_width(grid.get_grid_width()),
              grid_height(grid.get_grid_height()) {}

        using BaseSubscript = AccessorSubscript<Cell, GridAccessor, access_mode>;
        BaseSubscript operator[](uindex_t i) { return BaseSubscript(*this, i); }

        Cell const &operator[](sycl::id<2> id)
            requires(access_mode == sycl::access::mode::read)
        {
            return accessor[id[0] + halo_radius][id[1] + halo_radius];
        }

        Cell &operator[](sycl::id<2> id)
            requires(access_mode != sycl::access::mode::read)
        {
            return accessor[id[0] + halo_radius][id[1] + halo_radius];
        }

      private:
        sycl::host_accessor<Cell, 2, access_mode> accessor;
        uindex_t grid_width, grid_height;
    };

    void copy_from_buffer(sycl::buffer<Cell, 2> input_buffer) {
        if (input_buffer.get_range() !=
            sycl::range<2>(this->get_grid_width(), this->get_grid_height())) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::write_only};
        sycl::host_accessor input_ac{input_buffer, sycl::read_only};
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                grid_ac[c + halo_radius][r + halo_radius] = input_ac[c][r];
            }
        }
    }

    /**
     * \brief Copy the contents of the grid to a given buffer.
     *
     * This buffer has to exactly have the size of the grid, otherwise a `std::range_error` is
     * thrown.
     *
     * \param out_buffer The buffer to copy the cells to.
     * \throws std::out_of_range The buffer's size is not the same as the grid's size.
     */
    void copy_to_buffer(sycl::buffer<Cell, 2> output_buffer) {
        if (output_buffer.get_range() !=
            sycl::range<2>(this->get_grid_width(), this->get_grid_height())) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::read_only};
        sycl::host_accessor output_ac{output_buffer, sycl::write_only};
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                output_ac[c][r] = grid_ac[c + halo_radius][r + halo_radius];
            }
        }
    }

    /**
     * \brief Create a new grid that can be used as an output target.
     *
     * This grid will have the same range as the original grid and will use new buffers.
     *
     * \return The new grid.
     */
    Grid make_similar() const { return Grid(grid_width, grid_height); }

    uindex_t get_grid_width() const { return grid_width; }

    uindex_t get_grid_height() const { return grid_height; }

    /**
     * \brief Return the range of (central) tiles of the grid.
     *
     * This is not the range of a single tile nor is the range of the grid. It is the range of valid
     * arguments for \ref Grid.get_tile. For example, if the grid is 60 by 60 cells in size and
     * a tile is 32 by 32 cells in size, the tile range would be 2 by 2 tiles.
     *
     * \return The range of tiles of the grid.
     */
    GenericID<uindex_t> get_tile_range() const {
        return GenericID<uindex_t>(std::ceil(float(grid_width) / float(tile_width)),
                                   std::ceil(float(grid_height) / float(tile_height)));
    }

    template <typename in_pipe>
    void submit_read(sycl::queue &queue, uindex_t tile_c, uindex_t tile_r) {
        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::out_of_range("Tile index out of range!");
        }

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor grid_ac{grid_buffer, cgh, sycl::read_only};
            uindex_t grid_width = this->grid_width;
            uindex_t grid_height = this->grid_height;

            cgh.single_task([=]() {
                uindex_t start_c = tile_c * tile_width;
                uindex_t end_c = std::min((tile_c + 1) * tile_width, grid_width) + 2 * halo_radius;
                uindex_t start_r = tile_r * tile_height;
                uindex_t end_r =
                    std::min((tile_r + 1) * tile_height, grid_height) + 2 * halo_radius;

                for (uindex_t c = start_c; c < end_c; c++) {
                    for (uindex_t r = start_r; r < end_r; r++) {
                        in_pipe::write(grid_ac[c][r]);
                    }
                }
            });
        });
    }

    template <typename out_pipe>
    void submit_write(sycl::queue queue, uindex_t tile_c, uindex_t tile_r) {
        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::out_of_range("Tile index out of range!");
        }

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor grid_ac{grid_buffer, cgh, sycl::read_write};
            uindex_t grid_width = this->grid_width;
            uindex_t grid_height = this->grid_height;

            cgh.single_task([=]() {
                uindex_t start_c = tile_c * tile_width + halo_radius;
                uindex_t end_c = std::min((tile_c + 1) * tile_width, grid_width) + halo_radius;
                uindex_t start_r = tile_r * tile_height + halo_radius;
                uindex_t end_r = std::min((tile_r + 1) * tile_height, grid_height) + halo_radius;

                for (uindex_t c = start_c; c < end_c; c++) {
                    for (uindex_t r = start_r; r < end_r; r++) {
                        grid_ac[c][r] = out_pipe::read();
                    }
                }
            });
        });
    }

  private:
    sycl::buffer<Cell, 2> grid_buffer;
    uindex_t grid_width, grid_height;
};

} // namespace tiling
} // namespace stencil
