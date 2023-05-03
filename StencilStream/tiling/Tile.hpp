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
#include <optional>
#include <stdexcept>

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
 * \tparam Cell Cell value type.
 * \tparam width The number of columns of the tile.
 * \tparam height The number of rows of the tile.
 * \tparam halo_radius The radius (aka width and height) of the tile halo.
 */
template <typename Cell, uindex_t width, uindex_t height, uindex_t halo_radius, uindex_t word_size>
class Tile {
    static_assert(width > 2 * halo_radius);
    static_assert(height > 2 * halo_radius);

  public:
    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), word_size) / sizeof(Padded<Cell>);
    static constexpr uindex_t core_width = width - 2 * halo_radius;
    static constexpr uindex_t core_height = height - 2 * halo_radius;
    using IOWord = std::array<Padded<Cell>, word_length>;

    /**
     * \brief Create a new tile.
     *
     * Logically, the contents of the newly created tile are undefined since no memory resources are
     * allocated during construction and no initialization is done by the indexing operation.
     */
    Tile() : part{std::nullopt} {}

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
            return UID(halo_radius, halo_radius);
        case Part::NORTH_BORDER:
        case Part::SOUTH_BORDER:
            return UID(core_width, halo_radius);
        case Part::WEST_BORDER:
        case Part::EAST_BORDER:
            return UID(halo_radius, core_height);
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
    static cl::sycl::id<2> get_part_offset(Part part) {
        switch (part) {
        case Part::NORTH_WEST_CORNER:
            return cl::sycl::id<2>(0, 0);
        case Part::NORTH_BORDER:
            return cl::sycl::id<2>(halo_radius, 0);
        case Part::NORTH_EAST_CORNER:
            return cl::sycl::id<2>(width - halo_radius, 0);
        case Part::EAST_BORDER:
            return cl::sycl::id<2>(width - halo_radius, halo_radius);
        case Part::SOUTH_EAST_CORNER:
            return cl::sycl::id<2>(width - halo_radius, height - halo_radius);
        case Part::SOUTH_BORDER:
            return cl::sycl::id<2>(halo_radius, height - halo_radius);
        case Part::SOUTH_WEST_CORNER:
            return cl::sycl::id<2>(0, height - halo_radius);
        case Part::WEST_BORDER:
            return cl::sycl::id<2>(0, halo_radius);
        case Part::CORE:
            return cl::sycl::id<2>(halo_radius, halo_radius);
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }
    }

    static constexpr uindex_t n_cells_to_n_words(uindex_t n_cells) {
        return (n_cells / word_length) + (n_cells % word_length == 0 ? 0 : 1);
    }

    static uindex_t get_part_words(Part part) {
        UID range = get_part_range(part);
        uindex_t n_cells = range.c * range.r;
        return n_cells_to_n_words(n_cells);
    }

    static constexpr uindex_t max_n_cells() {
        uindex_t lhs = halo_radius >= width ? halo_radius : width;
        uindex_t rhs = halo_radius >= height ? halo_radius : height;
        return lhs * rhs;
    }

    static constexpr uindex_t max_n_words() { return n_cells_to_n_words(max_n_cells()); }

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
    cl::sycl::buffer<IOWord, 1> operator[](Part tile_part) {
        uindex_t part_column, part_row;
        switch (tile_part) {
        case Part::NORTH_WEST_CORNER:
            part_column = 0;
            part_row = 0;
            break;
        case Part::NORTH_BORDER:
            part_column = 1;
            part_row = 0;
            break;
        case Part::NORTH_EAST_CORNER:
            part_column = 2;
            part_row = 0;
            break;
        case Part::EAST_BORDER:
            part_column = 2;
            part_row = 1;
            break;
        case Part::SOUTH_EAST_CORNER:
            part_column = 2;
            part_row = 2;
            break;
        case Part::SOUTH_BORDER:
            part_column = 1;
            part_row = 2;
            break;
        case Part::SOUTH_WEST_CORNER:
            part_column = 0;
            part_row = 2;
            break;
        case Part::WEST_BORDER:
            part_column = 0;
            part_row = 1;
            break;
        case Part::CORE:
            part_column = 1;
            part_row = 1;
            break;
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }

        if (!part[part_column][part_row].has_value()) {
            cl::sycl::buffer<IOWord, 1> new_part = cl::sycl::range<1>(get_part_words(tile_part));
            part[part_column][part_row] = new_part;
        }
        return *part[part_column][part_row];
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
    void copy_from(cl::sycl::buffer<Cell, 2> buffer, cl::sycl::id<2> offset) {
        auto accessor = buffer.template get_access<cl::sycl::access::mode::read_write>();
        copy_part(accessor, Part::NORTH_WEST_CORNER, offset, true);
        copy_part(accessor, Part::NORTH_BORDER, offset, true);
        copy_part(accessor, Part::NORTH_EAST_CORNER, offset, true);
        copy_part(accessor, Part::EAST_BORDER, offset, true);
        copy_part(accessor, Part::SOUTH_EAST_CORNER, offset, true);
        copy_part(accessor, Part::SOUTH_BORDER, offset, true);
        copy_part(accessor, Part::SOUTH_WEST_CORNER, offset, true);
        copy_part(accessor, Part::WEST_BORDER, offset, true);
        copy_part(accessor, Part::CORE, offset, true);
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
    void copy_to(cl::sycl::buffer<Cell, 2> buffer, cl::sycl::id<2> offset) {
        auto accessor = buffer.template get_access<cl::sycl::access::mode::read_write>();
        copy_part(accessor, Part::NORTH_WEST_CORNER, offset, false);
        copy_part(accessor, Part::NORTH_BORDER, offset, false);
        copy_part(accessor, Part::NORTH_EAST_CORNER, offset, false);
        copy_part(accessor, Part::EAST_BORDER, offset, false);
        copy_part(accessor, Part::SOUTH_EAST_CORNER, offset, false);
        copy_part(accessor, Part::SOUTH_BORDER, offset, false);
        copy_part(accessor, Part::SOUTH_WEST_CORNER, offset, false);
        copy_part(accessor, Part::WEST_BORDER, offset, false);
        copy_part(accessor, Part::CORE, offset, false);
    }

  private:
    /**
     * \brief Helper function to copy a part to or from a buffer.
     */
    void copy_part(cl::sycl::accessor<Cell, 2, cl::sycl::access::mode::read_write,
                                      cl::sycl::access::target::host_buffer>
                       accessor,
                   Part part, cl::sycl::id<2> global_offset, bool buffer_to_part) {
        cl::sycl::id<2> offset = global_offset + get_part_offset(part);
        if (offset[0] >= accessor.get_range()[0] || offset[1] >= accessor.get_range()[1]) {
            // Nothing to do here. There is no data in the buffer for this part.
            return;
        }

        auto part_ac = (*this)[part].template get_access<cl::sycl::access::mode::read_write>();
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

    std::optional<cl::sycl::buffer<IOWord, 1>> part[3][3];
};

} // namespace tiling
} // namespace stencil