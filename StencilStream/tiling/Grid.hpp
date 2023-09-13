/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../Concepts.hpp"
#include "../GenericID.hpp"
#include "Tile.hpp"
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
          uindex_t halo_radius = 1, uindex_t word_size = 64>
class Grid {
  public:
    static_assert(2 * halo_radius < tile_height && 2 * halo_radius < tile_width);
    static constexpr uindex_t core_height = tile_height - 2 * halo_radius;
    static constexpr uindex_t core_width = tile_width - 2 * halo_radius;

    using Tile = Tile<Cell, tile_width, tile_height, halo_radius, word_size>;

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
        : tiles(), grid_width(grid_width), grid_height(grid_height) {
        allocate_tiles();
    }

    Grid(cl::sycl::buffer<Cell, 2> input_buffer)
        : tiles(), grid_width(input_buffer.get_range()[0]),
          grid_height(input_buffer.get_range()[1]) {
        copy_from_buffer(input_buffer);
    }

    template <cl::sycl::access::mode access_mode> class GridAccessor {
      public:
        using TileAccessor = typename Tile::template TileAccessor<access_mode>;

        GridAccessor(std::vector<std::vector<TileAccessor>> tile_acs, uindex_t grid_width,
                     uindex_t grid_height)
            : tile_acs(tile_acs), grid_width(grid_width), grid_height(grid_height) {}

        Cell get(uindex_t c, uindex_t r) const {
            return tile_acs[c / tile_width][r / tile_height].get(c % tile_width, r % tile_height);
        }

        void set(uindex_t c, uindex_t r, Cell cell) {
            tile_acs[c / tile_width][r / tile_height].set(c % tile_width, r % tile_height, cell);
        }

      private:
        std::vector<std::vector<TileAccessor>> tile_acs;
        uindex_t grid_width, grid_height;
    };

    template <cl::sycl::access::mode access_mode> GridAccessor<access_mode> get_access() {
        using GridAC = GridAccessor<access_mode>;
        using TileAC = GridAC::TileAccessor;

        std::vector<std::vector<TileAC>> tile_acs;
        for (uindex_t tile_c = 0; tile_c < get_tile_range().c; tile_c++) {
            tile_acs.push_back(std::vector<TileAC>());
            for (uindex_t tile_r = 0; tile_r < get_tile_range().r; tile_r++) {
                auto next_ac = get_tile(tile_c, tile_r).template get_access<access_mode>();
                tile_acs.back().push_back(next_ac);
            }
        }

        return GridAC(tile_acs, grid_width, grid_height);
    }

    void copy_from_buffer(cl::sycl::buffer<Cell, 2> input_buffer) {
        if (input_buffer.get_range() !=
            cl::sycl::range<2>(this->get_grid_width(), this->get_grid_height())) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        allocate_tiles();

        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++) {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++) {
                cl::sycl::id<2> offset((tile_column - 1) * tile_width,
                                       (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_from(input_buffer, offset);
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
     * \throws std::range_error The buffer's size is not the same as the grid's size.
     */
    void copy_to_buffer(cl::sycl::buffer<Cell, 2> output_buffer) {
        if (output_buffer.get_range() !=
            cl::sycl::range<2>(this->get_grid_width(), this->get_grid_height())) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++) {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++) {
                cl::sycl::id<2> offset((tile_column - 1) * tile_width,
                                       (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_to(output_buffer, offset);
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
        return GenericID<uindex_t>(tiles.size() - 2, tiles[0].size() - 2);
    }

    /**
     * \brief Get the tile at the given index.
     *
     * \param tile_id The id of the tile to return.
     * \return The tile.
     * \throws std::out_of_range Thrown if the tile id is outside the range of tiles, as returned by
     * \ref Grid.get_tile_range.
     */
    Tile &get_tile(index_t tile_c, index_t tile_r) { return tiles.at(tile_c + 1).at(tile_r + 1); }

    template <typename in_pipe>
    void submit_read(cl::sycl::queue queue, index_t tile_c, index_t tile_r) {
        using feed_in_pipe_0 = cl::sycl::pipe<class feed_in_pipe_0_id, Cell>;
        using feed_in_pipe_1 = cl::sycl::pipe<class feed_in_pipe_1_id, Cell>;
        using feed_in_pipe_2 = cl::sycl::pipe<class feed_in_pipe_2_id, Cell>;
        using feed_in_pipe_3 = cl::sycl::pipe<class feed_in_pipe_3_id, Cell>;
        using feed_in_pipe_4 = cl::sycl::pipe<class feed_in_pipe_4_id, Cell>;

        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::range_error("Tile ID out of range!");
        }

        auto part_widths = get_part_widths(tile_c);

        get_tile(tile_c - 1, tile_r - 1)
            .template submit_read_part<feed_in_pipe_0>(queue, Tile::Part::SOUTH_EAST_CORNER,
                                                       part_widths[0], halo_radius, 0, 0);
        get_tile(tile_c, tile_r - 1)
            .template submit_read_part<feed_in_pipe_0>(queue, Tile::Part::SOUTH_WEST_CORNER,
                                                       part_widths[1], halo_radius, 0, 0);
        get_tile(tile_c, tile_r - 1)
            .template submit_read_part<feed_in_pipe_0>(queue, Tile::Part::SOUTH_BORDER,
                                                       part_widths[2], halo_radius, 0, 0);
        get_tile(tile_c, tile_r - 1)
            .template submit_read_part<feed_in_pipe_0>(queue, Tile::Part::SOUTH_EAST_CORNER,
                                                       part_widths[3], halo_radius, 0, 0);
        get_tile(tile_c + 1, tile_r - 1)
            .template submit_read_part<feed_in_pipe_0>(queue, Tile::Part::SOUTH_WEST_CORNER,
                                                       part_widths[4], halo_radius, 0, 0);

        get_tile(tile_c - 1, tile_r)
            .template submit_read_part<feed_in_pipe_1>(queue, Tile::Part::NORTH_EAST_CORNER,
                                                       part_widths[0], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_1>(queue, Tile::Part::NORTH_WEST_CORNER,
                                                       part_widths[1], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_1>(queue, Tile::Part::NORTH_BORDER,
                                                       part_widths[2], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_1>(queue, Tile::Part::NORTH_EAST_CORNER,
                                                       part_widths[3], halo_radius, 0, 0);
        get_tile(tile_c + 1, tile_r)
            .template submit_read_part<feed_in_pipe_1>(queue, Tile::Part::NORTH_WEST_CORNER,
                                                       part_widths[4], halo_radius, 0, 0);

        get_tile(tile_c - 1, tile_r)
            .template submit_read_part<feed_in_pipe_2>(queue, Tile::Part::EAST_BORDER,
                                                       part_widths[0],
                                                       tile_height - 2 * halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_2>(queue, Tile::Part::WEST_BORDER,
                                                       part_widths[1],
                                                       tile_height - 2 * halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_2>(queue, Tile::Part::CORE, part_widths[2],
                                                       tile_height - 2 * halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_2>(queue, Tile::Part::EAST_BORDER,
                                                       part_widths[3],
                                                       tile_height - 2 * halo_radius, 0, 0);
        get_tile(tile_c + 1, tile_r)
            .template submit_read_part<feed_in_pipe_2>(queue, Tile::Part::WEST_BORDER,
                                                       part_widths[4],
                                                       tile_height - 2 * halo_radius, 0, 0);

        get_tile(tile_c - 1, tile_r)
            .template submit_read_part<feed_in_pipe_3>(queue, Tile::Part::SOUTH_EAST_CORNER,
                                                       part_widths[0], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_3>(queue, Tile::Part::SOUTH_WEST_CORNER,
                                                       part_widths[1], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_3>(queue, Tile::Part::SOUTH_BORDER,
                                                       part_widths[2], halo_radius, 0, 0);
        get_tile(tile_c, tile_r)
            .template submit_read_part<feed_in_pipe_3>(queue, Tile::Part::SOUTH_EAST_CORNER,
                                                       part_widths[3], halo_radius, 0, 0);
        get_tile(tile_c + 1, tile_r)
            .template submit_read_part<feed_in_pipe_3>(queue, Tile::Part::SOUTH_WEST_CORNER,
                                                       part_widths[4], halo_radius, 0, 0);

        get_tile(tile_c - 1, tile_r + 1)
            .template submit_read_part<feed_in_pipe_4>(queue, Tile::Part::NORTH_EAST_CORNER,
                                                       part_widths[0], halo_radius, 0, 0);
        get_tile(tile_c, tile_r + 1)
            .template submit_read_part<feed_in_pipe_4>(queue, Tile::Part::NORTH_WEST_CORNER,
                                                       part_widths[1], halo_radius, 0, 0);
        get_tile(tile_c, tile_r + 1)
            .template submit_read_part<feed_in_pipe_4>(queue, Tile::Part::NORTH_BORDER,
                                                       part_widths[2], halo_radius, 0, 0);
        get_tile(tile_c, tile_r + 1)
            .template submit_read_part<feed_in_pipe_4>(queue, Tile::Part::NORTH_EAST_CORNER,
                                                       part_widths[3], halo_radius, 0, 0);
        get_tile(tile_c + 1, tile_r + 1)
            .template submit_read_part<feed_in_pipe_4>(queue, Tile::Part::NORTH_WEST_CORNER,
                                                       part_widths[4], halo_radius, 0, 0);

        queue.submit([&](cl::sycl::handler &cgh) {
            uindex_t n_inner_columns = part_widths[1] + part_widths[2] + part_widths[3];

            cgh.single_task([=]() {
                constexpr unsigned long bits_width = std::bit_width(2 * halo_radius + tile_width);
                constexpr unsigned long bits_height = std::bit_width(2 * halo_radius + tile_height);
                using uindex_width_t = ac_int<bits_width, false>;
                using uindex_height_t = ac_int<bits_height, false>;

                [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                 c <
                                                 uindex_width_t(2 * halo_radius + n_inner_columns);
                                                 c++) {
                    for (uindex_height_t r = 0; r < uindex_height_t(2 * halo_radius + tile_height);
                         r++) {
                        Cell value;
                        if (r < uindex_height_t(halo_radius)) {
                            value = feed_in_pipe_0::read();
                        } else if (r < uindex_height_t(2 * halo_radius)) {
                            value = feed_in_pipe_1::read();
                        } else if (r < uindex_height_t(2 * halo_radius + core_height)) {
                            value = feed_in_pipe_2::read();
                        } else if (r < uindex_height_t(3 * halo_radius + core_height)) {
                            value = feed_in_pipe_3::read();
                        } else {
                            value = feed_in_pipe_4::read();
                        }
                        in_pipe::write(value);
                    }
                }
            });
        });
    }

    template <typename out_pipe>
    void submit_write(cl::sycl::queue queue, index_t tile_c, index_t tile_r) {
        using feed_out_pipe_0 = cl::sycl::pipe<class feed_out_pipe_0_id, Cell>;
        using feed_out_pipe_1 = cl::sycl::pipe<class feed_out_pipe_1_id, Cell>;
        using feed_out_pipe_2 = cl::sycl::pipe<class feed_out_pipe_2_id, Cell>;

        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::range_error("Tile ID out of range!");
        }

        auto part_widths = get_part_widths(tile_c);

        queue.submit([&](cl::sycl::handler &cgh) {
            uindex_t n_inner_columns = part_widths[1] + part_widths[2] + part_widths[3];

            cgh.single_task([=]() {
                constexpr unsigned long bits_width = std::bit_width(tile_width);
                constexpr unsigned long bits_height = std::bit_width(tile_height);
                using uindex_width_t = ac_int<bits_width, false>;
                using uindex_height_t = ac_int<bits_height, false>;

                [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                 c < uindex_width_t(n_inner_columns); c++) {
                    for (uindex_height_t r = 0; r < uindex_height_t(tile_height); r++) {
                        Cell value = out_pipe::read();
                        if (r < uindex_height_t(halo_radius)) {
                            feed_out_pipe_0::write(value);
                        } else if (r < uindex_height_t(tile_height - halo_radius)) {
                            feed_out_pipe_1::write(value);
                        } else {
                            feed_out_pipe_2::write(value);
                        }
                    }
                }
            });
        });

        Tile &tile = get_tile(tile_c, tile_r);
        tile.template submit_write_part<feed_out_pipe_0>(queue, Tile::Part::NORTH_WEST_CORNER,
                                                         part_widths[1], halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_0>(queue, Tile::Part::NORTH_BORDER,
                                                         part_widths[2], halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_0>(queue, Tile::Part::NORTH_EAST_CORNER,
                                                         part_widths[3], halo_radius, 0, 0);

        tile.template submit_write_part<feed_out_pipe_1>(
            queue, Tile::Part::WEST_BORDER, part_widths[1], tile_height - 2 * halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_1>(queue, Tile::Part::CORE, part_widths[2],
                                                         tile_height - 2 * halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_1>(
            queue, Tile::Part::EAST_BORDER, part_widths[3], tile_height - 2 * halo_radius, 0, 0);

        tile.template submit_write_part<feed_out_pipe_2>(queue, Tile::Part::SOUTH_WEST_CORNER,
                                                         part_widths[1], halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_2>(queue, Tile::Part::SOUTH_BORDER,
                                                         part_widths[2], halo_radius, 0, 0);
        tile.template submit_write_part<feed_out_pipe_2>(queue, Tile::Part::SOUTH_EAST_CORNER,
                                                         part_widths[3], halo_radius, 0, 0);
    }

  private:
    std::array<uindex_t, 5> get_part_widths(index_t tile_c) const {
        uindex_t column_offset = tile_c * tile_width;
        uindex_t n_inner_columns = std::min(tile_width, grid_width - column_offset);

        std::array<uindex_t, 5> part_widths;
        part_widths[0] = halo_radius;

        part_widths[1] = std::min(n_inner_columns, halo_radius);

        if (n_inner_columns > halo_radius) {
            part_widths[2] = std::min(n_inner_columns - halo_radius, Tile::core_width);
        } else {
            part_widths[2] = 0;
        }

        if (n_inner_columns > halo_radius + Tile::core_width) {
            part_widths[3] =
                std::min(n_inner_columns - halo_radius - Tile::core_width, halo_radius);
        } else {
            part_widths[3] = 0;
        }

        part_widths[4] = halo_radius;

        assert(part_widths[1] + part_widths[2] + part_widths[3] == n_inner_columns);

        return part_widths;
    }

    void allocate_tiles() {
        tiles.clear();

        uindex_t n_tile_columns = grid_width / tile_width;
        if (grid_width % tile_width != 0) {
            n_tile_columns++;
        }
        uindex_t n_tile_rows = grid_height / tile_height;
        if (grid_height % tile_height != 0) {
            n_tile_rows++;
        }

        tiles.reserve(n_tile_columns + 2);
        for (uindex_t i_column = 0; i_column < n_tile_columns + 2; i_column++) {
            std::vector<Tile> column;
            column.reserve(n_tile_rows + 2);
            for (uindex_t i_row = 0; i_row < n_tile_rows + 2; i_row++) {
                column.push_back(Tile());
            }
            tiles.push_back(column);
        }
    }

    std::vector<std::vector<Tile>> tiles;
    uindex_t grid_width, grid_height;
};

} // namespace tiling
} // namespace stencil