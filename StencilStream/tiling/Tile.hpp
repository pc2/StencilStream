/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <bit>
#include <optional>
#include <stdexcept>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace stencil {
namespace tiling {

struct TileParameters {
    uindex_t width = 1024;
    uindex_t height = 1024;
    uindex_t halo_radius = 1;
    uindex_t word_size = 64;
};

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
 * \tparam Cell Cell value type.
 * \tparam width The number of columns of the tile.
 * \tparam height The number of rows of the tile.
 * \tparam halo_radius The radius (aka width and height) of the tile halo.
 */
template <typename Cell, TileParameters params> class Tile {
    static_assert(params.width > 2 * params.halo_radius);
    static_assert(params.height > 2 * params.halo_radius);

  public:
    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), params.word_size) / sizeof(Padded<Cell>);
    static constexpr uindex_t core_width = params.width - 2 * params.halo_radius;
    static constexpr uindex_t core_height = params.height - 2 * params.halo_radius;
    using IOWord = std::array<Padded<Cell>, word_length>;

    /**
     * \brief Enumeration to address individual parts of the tile.
     */
    enum class Part {
        NORTH_WEST_CORNER,
        NORTH_BORDER,
        NORTH_EAST_CORNER,
        EAST_BORDER,
        SOUTH_EAST_CORNER,
        SOUTH_BORDER,
        SOUTH_WEST_CORNER,
        WEST_BORDER,
        CORE,
    };

    /**
     * \brief Array with all \ref Part variants to allow iteration over them.
     */
    static constexpr Part all_parts[] = {
        Part::NORTH_WEST_CORNER, Part::NORTH_BORDER,      Part::NORTH_EAST_CORNER,
        Part::EAST_BORDER,       Part::SOUTH_EAST_CORNER, Part::SOUTH_BORDER,
        Part::SOUTH_WEST_CORNER, Part::WEST_BORDER,       Part::CORE,
    };

    /**
     * \brief Create a new tile.
     *
     * Logically, the contents of the newly created tile are undefined since no memory resources are
     * allocated during construction and no initialization is done by the indexing operation.
     */
    Tile()
        : part_buffer{
              {sycl::range<1>(get_part_words(Part::NORTH_WEST_CORNER)),
               sycl::range<1>(get_part_words(Part::WEST_BORDER)),
               sycl::range<1>(get_part_words(Part::SOUTH_WEST_CORNER))},
              {sycl::range<1>(get_part_words(Part::NORTH_BORDER)),
               sycl::range<1>(get_part_words(Part::CORE)),
               sycl::range<1>(get_part_words(Part::SOUTH_BORDER))},
              {sycl::range<1>(get_part_words(Part::NORTH_EAST_CORNER)),
               sycl::range<1>(get_part_words(Part::EAST_BORDER)),
               sycl::range<1>(get_part_words(Part::SOUTH_EAST_CORNER))},
          } {}

    static uindex_t get_part_words_in_column(Part part) {
        UID range = get_part_range(part);
        return n_cells_to_n_words(range.r, word_length);
    }

    static uindex_t get_part_words(Part part) {
        UID range = get_part_range(part);
        return range.c * get_part_words_in_column(part);
    }

    static UID get_part_id(Part tile_part) {
        switch (tile_part) {
        case Part::NORTH_WEST_CORNER:
            return UID(0, 0);
        case Part::NORTH_BORDER:
            return UID(1, 0);
        case Part::NORTH_EAST_CORNER:
            return UID(2, 0);
        case Part::EAST_BORDER:
            return UID(2, 1);
        case Part::SOUTH_EAST_CORNER:
            return UID(2, 2);
        case Part::SOUTH_BORDER:
            return UID(1, 2);
        case Part::SOUTH_WEST_CORNER:
            return UID(0, 2);
        case Part::WEST_BORDER:
            return UID(0, 1);
        case Part::CORE:
            return UID(1, 1);
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }
    }

    /**
     * \brief Calculate the range of a given Part.
     *
     * \param part The part to calculate the range for.
     * \return The range of the part, used for example to allocate it.
     */
    static UID get_part_range(Part part) {
        switch (part) {
        case Part::NORTH_WEST_CORNER:
        case Part::SOUTH_WEST_CORNER:
        case Part::SOUTH_EAST_CORNER:
        case Part::NORTH_EAST_CORNER:
            return UID(params.halo_radius, params.halo_radius);
        case Part::NORTH_BORDER:
        case Part::SOUTH_BORDER:
            return UID(core_width, params.halo_radius);
        case Part::WEST_BORDER:
        case Part::EAST_BORDER:
            return UID(params.halo_radius, core_height);
        case Part::CORE:
            return UID(core_width, core_height);
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }
    }

    /**
     * \brief Calculate the index offset of a given part relative to the north-western corner of the
     * tile.
     *
     * \param part The part to calculate the offset for.
     * \return The offset of the part.
     */
    static sycl::id<2> get_part_offset(Part part) {
        switch (part) {
        case Part::NORTH_WEST_CORNER:
            return sycl::id<2>(0, 0);
        case Part::NORTH_BORDER:
            return sycl::id<2>(params.halo_radius, 0);
        case Part::NORTH_EAST_CORNER:
            return sycl::id<2>(params.width - params.halo_radius, 0);
        case Part::EAST_BORDER:
            return sycl::id<2>(params.width - params.halo_radius, params.halo_radius);
        case Part::SOUTH_EAST_CORNER:
            return sycl::id<2>(params.width - params.halo_radius,
                               params.height - params.halo_radius);
        case Part::SOUTH_BORDER:
            return sycl::id<2>(params.halo_radius, params.height - params.halo_radius);
        case Part::SOUTH_WEST_CORNER:
            return sycl::id<2>(0, params.height - params.halo_radius);
        case Part::WEST_BORDER:
            return sycl::id<2>(0, params.halo_radius);
        case Part::CORE:
            return sycl::id<2>(params.halo_radius, params.halo_radius);
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }
    }

    template <sycl::access::mode access_mode = sycl::access::mode::read_write> class TileAccessor {
      public:
        TileAccessor(Tile &tile) : part_ac() {
            for (uindex_t c = 0; c < 3; c++) {
                for (uindex_t r = 0; r < 3; r++) {
                    part_ac[c][r] = tile.part_buffer[c][r];
                }
            }
        }

        Cell get(uindex_t c, uindex_t r) const {
            std::array<uindex_t, 4> i = transform_indices(c, r);
            return part_ac[i[0]][i[1]][i[2]][i[3]].value;
        }

        void set(uindex_t c, uindex_t r, Cell cell) {
            std::array<uindex_t, 4> i = transform_indices(c, r);
            part_ac[i[0]][i[1]][i[2]][i[3]].value = cell;
        }

        static std::array<uindex_t, 4> transform_indices(uindex_t c, uindex_t r) {
            uindex_t part_c;
            if (c < params.halo_radius) {
                part_c = 0;
            } else if (c < params.width - params.halo_radius) {
                part_c = 1;
                c -= params.halo_radius;
            } else {
                part_c = 2;
                c -= params.width - params.halo_radius;
            }
            uindex_t part_r, part_words_per_column;
            if (r < params.halo_radius) {
                part_r = 0;
                part_words_per_column = n_cells_to_n_words(params.halo_radius, word_length);
            } else if (r < params.height - params.halo_radius) {
                part_r = 1;
                part_words_per_column =
                    n_cells_to_n_words(params.height - 2 * params.halo_radius, word_length);
                r -= params.halo_radius;
            } else {
                part_r = 2;
                part_words_per_column = n_cells_to_n_words(params.halo_radius, word_length);
                r -= params.height - params.halo_radius;
            }
            uindex_t word_i = (c * part_words_per_column) + (r / word_length);
            uindex_t element_i = r % word_length;
            return {part_c, part_r, word_i, element_i};
        }

      private:
        sycl::host_accessor<IOWord, 1, access_mode> part_ac[3][3];
    };

    template <sycl::access::mode access_mode> TileAccessor<access_mode> get_access() {
        return TileAccessor<access_mode>(*this);
    }

    /**
     * \brief Return the buffer with the contents of the given part.
     *
     * If the part has not been accessed before, it will allocate the part's buffer. Note however
     * that this method does not initialize the buffer. Please also note that the buffer is
     * word-aligned: The height of the returned buffer (the second value of the range) is always
     * `word_length` and the width is big enough to store all required cells of the part. For more
     * information, read about \ref wordalignment.
     *
     * \param tile_part The part to access.
     * \return The buffer of the part.
     */
    sycl::buffer<IOWord, 1> get_part_buffer(Part tile_part) {
        UID part_id = get_part_id(tile_part);
        return part_buffer[part_id.c][part_id.r];
    }

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
    void copy_from(sycl::buffer<Cell, 2> buffer, sycl::id<2> offset) {
        sycl::host_accessor ac(buffer, sycl::read_write);
        for (Part part : all_parts) {
            copy_part(ac, part, offset, true);
        }
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
    void copy_to(sycl::buffer<Cell, 2> buffer, sycl::id<2> offset) {
        sycl::host_accessor ac(buffer, sycl::read_write);
        for (Part part : all_parts) {
            copy_part(ac, part, offset, false);
        }
    }

    template <typename in_pipe>
    void submit_read_part(cl::sycl::queue queue, Part part, uindex_t n_columns) {
        if (n_columns == 0) {
            return;
        }

        queue.submit([&](cl::sycl::handler &cgh) {
            auto part_buffer = get_part_buffer(part);
            sycl::accessor ac(part_buffer, cgh, sycl::read_only);
            UID range = get_part_range(part);
            uindex_t n_cells = n_columns * range.r;

            cgh.single_task([=]() {
                [[intel::fpga_register]] IOWord cache;
                uindex_t word_i = 0;
                uindex_t cell_i = word_length;

                for (uindex_t i = 0; i < n_cells; i++) {
                    if (cell_i == word_length) {
                        cache = ac[word_i];
                        word_i++;
                        cell_i = 0;
                    }
                    in_pipe::write(cache[cell_i].value);
                    cell_i++;
                }
            });
        });
    }

    template <typename out_pipe>
    void submit_write_part(cl::sycl::queue queue, Part part, uindex_t n_columns) {
        if (n_columns == 0) {
            return;
        }

        queue.submit([&](cl::sycl::handler &cgh) {
            auto part_buffer = get_part_buffer(part);
            sycl::accessor ac(part_buffer, cgh, sycl::write_only);
            UID range = get_part_range(part);
            uindex_t n_cells = n_columns * range.r;

            cgh.single_task([=]() {
                [[intel::fpga_memory]] IOWord cache;
                uindex_t word_i = 0;
                uindex_t cell_i = 0;

                for (uindex_t i = 0; i < n_cells; i++) {
                    if (cell_i == word_length) {
                        ac[word_i] = cache;
                        word_i++;
                        cell_i = 0;
                    }
                    cache[cell_i].value = out_pipe::read();
                    cell_i++;
                }

                if (cell_i != 0) {
                    ac[word_i] = cache;
                }
            });
        });
    }

  private:
    /**
     * \brief Helper function to copy a part to or from a buffer.
     */
    void copy_part(sycl::host_accessor<Cell, 2, sycl::access::mode::read_write> accessor, Part part,
                   sycl::id<2> global_offset, bool buffer_to_part) {
        sycl::id<2> offset = global_offset + get_part_offset(part);
        if (offset[0] >= accessor.get_range()[0] || offset[1] >= accessor.get_range()[1]) {
            // Nothing to do here. There is no data in the buffer for this part.
            return;
        }

        auto part_buffer = get_part_buffer(part);
        sycl::host_accessor part_ac(part_buffer, sycl::read_write);
        uindex_t part_width = get_part_range(part).c;
        uindex_t part_height = get_part_range(part).r;

        for (uindex_t c = 0; c < part_width; c++) {
            for (uindex_t r = 0; r < part_height; r++) {
                uindex_t word_i = (c * part_height + r) / word_length;
                uindex_t cell_i = (c * part_height + r) % word_length;
                uindex_t global_c = offset[0] + c;
                uindex_t global_r = offset[1] + r;

                if (global_c >= accessor.get_range()[0] || global_r >= accessor.get_range()[1]) {
                    continue;
                }

                if (buffer_to_part) {
                    part_ac[word_i][cell_i].value = accessor[global_c][global_r];
                } else {
                    accessor[global_c][global_r] = part_ac[word_i][cell_i].value;
                }
            }
        }
    }

    sycl::buffer<IOWord, 1> part_buffer[3][3];
};

} // namespace tiling
} // namespace stencil