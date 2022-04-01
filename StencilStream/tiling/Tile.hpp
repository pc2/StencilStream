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
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../Padded.hpp"
#include <CL/sycl.hpp>
#include <optional>
#include <stdexcept>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace stencil {
namespace tiling {

/**
 * \brief A rectangular container of cells with a static size
 *
 * StencilStream tiles the grid and the \ref ExecutionKernel works with those tiles: it
 * receives the content of a tile (together with it's halo) and emits the contents of a tile. A tile
 * is partitioned in four corner buffers, four border buffers and one core buffer. This is done to
 * provide easy access to the halo of a tile. Have a look at the \ref Architecture for more details
 * of the data layout.
 *
 * This tile manager supports copy operations to and from monolithic, user-supplied buffers, as well
 * as an index operation to access the individual parts.
 *
 * \tparam T Cell value type.
 * \tparam width The number of columns of the tile.
 * \tparam height The number of rows of the tile.
 * \tparam halo_radius The radius (aka width and height) of the tile halo.
 */
template <typename T, uindex_t max_width, uindex_t height, uindex_t halo_radius,
          uindex_t burst_size>
class Tile {
  public:
    static constexpr uindex_t burst_buffer_size = std::lcm(sizeof(Padded<T>), burst_size);
    static constexpr uindex_t burst_buffer_length = burst_buffer_size / sizeof(Padded<T>);
    using BurstBuffer = std::array<Padded<T>, burst_buffer_length>;

    static_assert(height > 2 * halo_radius);
    static constexpr uindex_t core_height = height - 2 * halo_radius;

    static constexpr uindex_t n_cells_to_n_bursts(uindex_t n_cells) {
        return (n_cells / burst_buffer_length) + (n_cells % burst_buffer_length == 0 ? 0 : 1);
    }

    static constexpr uindex_t max_n_cells_per_stripe() {
        return std::max(halo_radius, height) * max_width;
    }

    static constexpr uindex_t max_n_bursts_per_stripe() {
        return n_cells_to_n_bursts(max_n_cells_per_stripe());
    }

    static constexpr unsigned long bits_cell = std::bit_width(burst_buffer_length);
    using index_cell_t = ac_int<bits_cell + 1, true>;
    using uindex_cell_t = ac_int<bits_cell, false>;

    static constexpr unsigned long bits_burst = std::bit_width(max_n_bursts_per_stripe());
    using index_burst_t = ac_int<bits_burst + 1, true>;
    using uindex_burst_t = ac_int<bits_burst, false>;

    static constexpr unsigned long bits_2d = std::bit_width(max_n_cells_per_stripe());
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    /**
     * \brief Create a new tile.
     *
     * Logically, the contents of the newly created tile are undefined since no memory resources are
     * allocated during construction and no initialization is done by the indexing operation.
     */
    Tile(uindex_t width)
        : width(width), stripe{cl::sycl::range<1>(n_cells_to_n_bursts(halo_radius * width)),
                               cl::sycl::range<1>(n_cells_to_n_bursts(core_height * width)),
                               cl::sycl::range<1>(n_cells_to_n_bursts(halo_radius * width))} {
        assert(width >= halo_radius && width <= max_width);
    }

    uindex_t get_width() const { return width; }

    /**
     * \brief Enumeration to address individual stripes of the tile.
     */
    enum class Stripe {
        NORTH,
        CORE,
        SOUTH,
    };

    static constexpr std::array<Stripe, 3> all_stripes = {
        Stripe::NORTH,
        Stripe::CORE,
        Stripe::SOUTH,
    };

    static constexpr uindex_t get_stripe_height(Stripe stripe) {
        if (stripe == Stripe::CORE) {
            return core_height;
        } else {
            return halo_radius;
        }
    }

    static constexpr uindex_t stripe_to_index(Stripe stripe) {
        switch (stripe) {
        case Stripe::NORTH:
            return 0;
        case Stripe::CORE:
            return 1;
        case Stripe::SOUTH:
            return 2;
        }
    }

    enum class StripePart {
        HEADER,
        FULL,
        FOOTER,
    };

    static constexpr std::array<StripePart, 3> all_stripe_parts = {
        StripePart::HEADER,
        StripePart::FULL,
        StripePart::FOOTER,
    };

    /**
     * \brief Copy the contents of a buffer into the tile.
     *
     * This will take the buffer section `[offset[0] : min(grid_width, offset[0] + tile_width)) x
     * [offset[1] : min(grid_height, offset[1] + tile_height))` and copy it to the tile. This means
     * that if the grid does not contain enough cells to fill the whole tile, those missing cells
     * are left as-is.
     *
     * \param buffer The buffer to copy the data from.
     * \param offset The offset of the buffer section relative to the origin of the buffer.
     */
    void copy_from(cl::sycl::buffer<T, 2> buffer, cl::sycl::id<2> offset) {
        auto accessor = buffer.template get_access<cl::sycl::access::mode::read_write>();
        copy_stripe(accessor, 0, offset, true);
        copy_stripe(accessor, 1, offset, true);
        copy_stripe(accessor, 2, offset, true);
    }

    /**
     * \brief Copy the contents of the tile to a buffer.
     *
     * This will take the tile section `[0 : min(tile_width, grid_width - offset[0])) x [0 :
     * min(tile_height, grid_height - offset[1]))` and copy it to the buffer section `[offset[0] :
     * min(grid_width, offset[0] + tile_width)) x [offset[1] : min(grid_height, offset[1] +
     * tile_height))`. This means that if the tile plus the offset is wider or higher than the grid,
     * those superfluous cells are ignored.
     *
     * \param buffer The buffer to copy the data to.
     * \param offset The offset of the buffer section relative to the origin of the buffer.
     */
    void copy_to(cl::sycl::buffer<T, 2> buffer, cl::sycl::id<2> offset) {
        auto accessor = buffer.template get_access<cl::sycl::access::mode::read_write>();
        copy_stripe(accessor, 0, offset, false);
        copy_stripe(accessor, 1, offset, false);
        copy_stripe(accessor, 2, offset, false);
    }

    template <typename pipe>
    void submit_read(cl::sycl::queue fpga_queue, Stripe stripe, StripePart part) {
        fpga_queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = this->stripe[stripe_to_index(stripe)]
                          .template get_access<cl::sycl::access::mode::read>(cgh);

            uindex_t part_width;
            if (part == StripePart::FULL) {
                part_width = width;
            } else {
                part_width = halo_radius;
            }
            uindex_t part_height;
            if (stripe == Stripe::CORE) {
                part_height = core_height;
            } else {
                part_height = halo_radius;
            }
            uindex_2d_t n_cells = uindex_t(part_width * part_height);

            uindex_t starting_burst_i;
            uindex_t starting_cell_i;
            switch (part) {
            case StripePart::HEADER:
            case StripePart::FULL:
                starting_burst_i = 0;
                starting_cell_i = 0;
                break;
            case StripePart::FOOTER:
                starting_burst_i = (width - halo_radius) * part_height / burst_buffer_length;
                starting_cell_i = (width - halo_radius) * part_height % burst_buffer_length;
                break;
            }

            cgh.single_task<pipe>([=]() {
                [[intel::fpga_register]] BurstBuffer cache = ac[starting_burst_i];
                uindex_burst_t burst_i = starting_burst_i + 1;
                uindex_cell_t cell_i = starting_cell_i;

                for (uindex_2d_t i = 0; i < n_cells; i++) {
                    if (cell_i == uindex_cell_t(burst_buffer_length)) {
                        cache = ac[burst_i.to_uint64()];
                        burst_i++;
                        cell_i = 0;
                    }
                    pipe::write(cache[cell_i].value);
                    cell_i++;
                }
            });
        });
    }

    template <typename pipe> void submit_write(cl::sycl::queue fpga_queue, Stripe stripe) {
        fpga_queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = this->stripe[stripe_to_index(stripe)]
                          .template get_access<cl::sycl::access::mode::discard_write>(cgh);
            uindex_t stripe_height = get_stripe_height(stripe);
            uindex_2d_t n_cells = uindex_2d_t(width * stripe_height);

            cgh.single_task<pipe>([=]() {
                [[intel::fpga_memory]] BurstBuffer cache;
                uindex_burst_t burst_i = 0;
                uindex_cell_t cell_i = 0;

                for (uindex_2d_t i = 0; i < n_cells; i++) {
                    if (cell_i == uindex_cell_t(burst_buffer_length)) {
                        ac[burst_i.to_uint64()] = cache;
                        burst_i++;
                        cell_i = 0;
                    }
                    cache[cell_i].value = pipe::read();
                    cell_i++;
                }

                if (cell_i != 0) {
                    ac[burst_i.to_uint64()] = cache;
                }
            });
        });
    }

  private:
    /**
     * \brief Helper function to copy a stripe to or from a buffer.
     */
    void copy_stripe(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write,
                                        cl::sycl::access::target::host_buffer>
                         accessor,
                     uindex_t i_stripe, cl::sycl::id<2> global_offset, bool buffer_to_stripe) {
        uindex_t c_offset = global_offset[0];
        uindex_t r_offset = global_offset[1];
        if (i_stripe >= 1) {
            r_offset += halo_radius;
        }
        if (i_stripe == 2) {
            r_offset += core_height;
        }

        if (c_offset >= accessor.get_range()[0] || r_offset >= accessor.get_range()[1]) {
            // Nothing to do here. There is no data in the buffer for this stripe.
            return;
        }

        auto stripe_ac = stripe[i_stripe].template get_access<cl::sycl::access::mode::read_write>();
        uindex_t stripe_height = i_stripe == 1 ? core_height : halo_radius;

        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < stripe_height; r++) {
                uindex_t burst_i = (c * stripe_height + r) / burst_buffer_length;
                uindex_t cell_i = (c * stripe_height + r) % burst_buffer_length;
                uindex_t global_c = c_offset + c;
                uindex_t global_r = r_offset + r;

                if (global_c >= accessor.get_range()[0] || global_r >= accessor.get_range()[1]) {
                    continue;
                }

                if (buffer_to_stripe) {
                    stripe_ac[burst_i][cell_i].value = accessor[global_c][global_r];
                } else {
                    accessor[global_c][global_r] = stripe_ac[burst_i][cell_i].value;
                }
            }
        }
    }

    uindex_t width;
    cl::sycl::buffer<BurstBuffer, 1> stripe[3];
};

} // namespace tiling
} // namespace stencil