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
#include "../GenericID.hpp"
#include "Tile.hpp"
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/queue.hpp>
#include <bit>
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
 * \tparam T Cell value type.
 * \tparam max_tile_width The number of columns of a tile.
 * \tparam tile_height The number of rows of a tile.
 * \tparam halo_radius The radius (aka width and height) of the tile halo.
 */
template <typename T, uindex_t max_tile_width, uindex_t tile_height, uindex_t halo_radius,
          uindex_t burst_size = 64>
class Grid {
  public:
    static_assert(2 * halo_radius < tile_height);
    static constexpr uindex_t core_height = tile_height - 2 * halo_radius;

    using Tile = Tile<T, max_tile_width, tile_height, halo_radius, burst_size>;

    static constexpr uindex_t bits_width = std::bit_width(max_tile_width);
    static constexpr uindex_t bits_height = std::bit_width(tile_height);
    using uindex_width_t = ac_int<bits_width, false>;
    using uindex_height_t = ac_int<bits_height, false>;

    /**
     * \brief Create a grid with undefined contents.
     *
     * This constructor is used to create the output grid of a \ref ExecutionKernel
     * invocation. It's contents do not need to be initialized or copied from another buffer since
     * it will override cell values from the execution kernel anyway.
     *
     * \param width The number of columns of the grid.
     * \param height The number of rows of the grid.
     */
    Grid(uindex_t width, uindex_t height) : tiles(), grid_range(width, height) { allocate_tiles(); }

    /**
     * \brief Create a grid that contains the cells of a buffer.
     *
     * This will allocate enough resources to store the content of the buffer and than copy those
     * cells into the data layout of the grid.
     *
     * \param in_buffer The buffer to copy the cells from.
     */
    Grid(cl::sycl::buffer<T, 2> in_buffer) : tiles(), grid_range(in_buffer.get_range()) {
        copy_from(in_buffer);
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
    void copy_to(cl::sycl::buffer<T, 2> &out_buffer) {
        if (out_buffer.get_range() != grid_range) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++) {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++) {
                cl::sycl::id<2> offset((tile_column - 1) * max_tile_width,
                                       (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_to(out_buffer, offset);
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
    Grid make_output_grid() const {
        Grid output_grid(grid_range[0], grid_range[1]);

        return output_grid;
    }

    /**
     * \brief Return the range of (central) tiles of the grid.
     *
     * This is not the range of a single tile nor is the range of the grid. It is the range of valid
     * arguments for \ref Grid.get_tile. For example, if the grid is 60 by 60 cells in size and a
     * tile is 32 by 32 cells in size, the tile range would be 2 by 2 tiles.
     *
     * \return The range of tiles of the grid.
     */
    GenericID<uindex_t> get_tile_range() const {
        return GenericID<uindex_t>(tiles.size() - 2, tiles[0].size() - 2);
    }

    /**
     * \brief Return the range of the grid in cells.
     *
     * If the grid was constructed from a buffer, this will be the range of this buffer. If the grid
     * was constructed with width and height arguments, this will be this exact range.
     *
     * \return The range of cells of the grid.
     */
    GenericID<uindex_t> get_grid_range() const { return grid_range; }

    /**
     * \brief Get the tile at the given index.
     *
     * \param tile_id The id of the tile to return.
     * \return The tile.
     * \throws std::out_of_range Thrown if the tile id is outside the range of tiles, as returned by
     * \ref Grid.get_tile_range.
     */
    Tile &get_tile(ID tile_id) {
        uindex_t tile_c = tile_id.c + 1;
        uindex_t tile_r = tile_id.r + 1;
        return tiles.at(tile_c).at(tile_r);
    }

    template <typename in_pipe> void submit_read(cl::sycl::queue queue, ID tile_id) {
        using feed_in_pipe_0 = cl::sycl::pipe<class feed_in_pipe_0_id, T>;
        using feed_in_pipe_1 = cl::sycl::pipe<class feed_in_pipe_1_id, T>;
        using feed_in_pipe_2 = cl::sycl::pipe<class feed_in_pipe_2_id, T>;
        using feed_in_pipe_3 = cl::sycl::pipe<class feed_in_pipe_3_id, T>;
        using feed_in_pipe_4 = cl::sycl::pipe<class feed_in_pipe_4_id, T>;

        assert(tile_id.c >= 0 && tile_id.c < get_tile_range().c);
        assert(tile_id.r >= 0 && tile_id.r < get_tile_range().r);

        uindex_t tile_width = get_tile(tile_id).get_width();
        queue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task<class TilingMergeKernel>([=]() {
                [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                 c < uindex_width_t(2 * halo_radius + tile_width);
                                                 c++) {
                    for (uindex_height_t r = 0; r < uindex_height_t(2 * halo_radius + tile_height);
                         r++) {
                        T value;
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

        get_tile(ID(tile_id.c - 1, tile_id.r - 1))
            .template submit_read<feed_in_pipe_0>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::FOOTER);
        get_tile(ID(tile_id.c - 1, tile_id.r))
            .template submit_read<feed_in_pipe_1>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::FOOTER);
        get_tile(ID(tile_id.c - 1, tile_id.r))
            .template submit_read<feed_in_pipe_2>(queue, Tile::Stripe::CORE,
                                                  Tile::StripePart::FOOTER);
        get_tile(ID(tile_id.c - 1, tile_id.r))
            .template submit_read<feed_in_pipe_3>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::FOOTER);
        get_tile(ID(tile_id.c - 1, tile_id.r + 1))
            .template submit_read<feed_in_pipe_4>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::FOOTER);

        get_tile(ID(tile_id.c, tile_id.r - 1))
            .template submit_read<feed_in_pipe_0>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::FULL);
        get_tile(ID(tile_id.c, tile_id.r))
            .template submit_read<feed_in_pipe_1>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::FULL);
        get_tile(ID(tile_id.c, tile_id.r))
            .template submit_read<feed_in_pipe_2>(queue, Tile::Stripe::CORE,
                                                  Tile::StripePart::FULL);
        get_tile(ID(tile_id.c, tile_id.r))
            .template submit_read<feed_in_pipe_3>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::FULL);
        get_tile(ID(tile_id.c, tile_id.r + 1))
            .template submit_read<feed_in_pipe_4>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::FULL);

        get_tile(ID(tile_id.c + 1, tile_id.r - 1))
            .template submit_read<feed_in_pipe_0>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::HEADER);
        get_tile(ID(tile_id.c + 1, tile_id.r))
            .template submit_read<feed_in_pipe_1>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::HEADER);
        get_tile(ID(tile_id.c + 1, tile_id.r))
            .template submit_read<feed_in_pipe_2>(queue, Tile::Stripe::CORE,
                                                  Tile::StripePart::HEADER);
        get_tile(ID(tile_id.c + 1, tile_id.r))
            .template submit_read<feed_in_pipe_3>(queue, Tile::Stripe::SOUTH,
                                                  Tile::StripePart::HEADER);
        get_tile(ID(tile_id.c + 1, tile_id.r + 1))
            .template submit_read<feed_in_pipe_4>(queue, Tile::Stripe::NORTH,
                                                  Tile::StripePart::HEADER);
    }

    template <typename out_pipe> void submit_write(cl::sycl::queue queue, ID tile_id) {
        using feed_out_pipe_0 = cl::sycl::pipe<class feed_out_pipe_0_id, T>;
        using feed_out_pipe_1 = cl::sycl::pipe<class feed_out_pipe_1_id, T>;
        using feed_out_pipe_2 = cl::sycl::pipe<class feed_out_pipe_2_id, T>;

        assert(tile_id.c <= get_tile_range().c && tile_id.r <= get_tile_range().r);

        Tile &tile = get_tile(tile_id);
        uindex_t tile_width = tile.get_width();

        queue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task<class TilingForkKernel>([=]() {
                [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                 c < uindex_width_t(tile_width); c++) {
                    for (uindex_height_t r = 0; r < uindex_height_t(tile_height); r++) {
                        T value = out_pipe::read();
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

        tile.template submit_write<feed_out_pipe_0>(queue, Tile::Stripe::NORTH);
        tile.template submit_write<feed_out_pipe_1>(queue, Tile::Stripe::CORE);
        tile.template submit_write<feed_out_pipe_2>(queue, Tile::Stripe::SOUTH);
    }

  private:
    void copy_from(cl::sycl::buffer<T, 2> in_buffer) {
        if (in_buffer.get_range() != grid_range) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        allocate_tiles();

        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++) {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++) {
                cl::sycl::id<2> offset((tile_column - 1) * max_tile_width,
                                       (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_from(in_buffer, offset);
            }
        }
    }

    void allocate_tiles() {
        tiles.clear();

        uindex_t n_tile_columns = grid_range[0] / max_tile_width;
        if (grid_range[0] % max_tile_width != 0) {
            n_tile_columns++;
        }
        uindex_t n_tile_rows = grid_range[1] / tile_height;
        if (grid_range[1] % tile_height != 0) {
            n_tile_rows++;
        }

        tiles.reserve(n_tile_columns + 2);
        for (uindex_t i_column = 0; i_column < n_tile_columns + 2; i_column++) {
            std::vector<Tile> column;
            column.reserve(n_tile_rows + 2);
            for (uindex_t i_row = 0; i_row < n_tile_rows + 2; i_row++) {
                uindex_t tile_width;
                if (i_row == 0 || i_row == n_tile_rows + 1) {
                    tile_width = halo_radius;
                } else {
                    tile_width = std::min(max_tile_width,
                                          uindex_t(grid_range[0] - i_column * max_tile_width));
                }
                column.push_back(Tile(tile_width));
            }
            tiles.push_back(column);
        }
    }

    std::vector<std::vector<Tile>> tiles;
    cl::sycl::range<2> grid_range;
};

} // namespace tiling
} // namespace stencil