/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "Helpers.hpp"
#include "Index.hpp"
#include <CL/sycl.hpp>
#include <optional>
#include <stdexcept>

namespace stencil
{
template <typename T, uindex_t width, uindex_t height, uindex_t halo_radius, uindex_t burst_length>
class Tile
{
    static_assert(width > 2 * halo_radius);
    static_assert(height > 2 * halo_radius);

public:
    Tile() : part{std::nullopt} {}

    enum class Part
    {
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

    static constexpr Part all_parts[] = {
        Part::NORTH_WEST_CORNER,
        Part::NORTH_BORDER,
        Part::NORTH_EAST_CORNER,
        Part::EAST_BORDER,
        Part::SOUTH_EAST_CORNER,
        Part::SOUTH_BORDER,
        Part::SOUTH_WEST_CORNER,
        Part::WEST_BORDER,
        Part::CORE,
    };

    static cl::sycl::range<2> get_part_range(Part part)
    {
        switch (part)
        {
        case Part::NORTH_WEST_CORNER:
        case Part::SOUTH_WEST_CORNER:
        case Part::SOUTH_EAST_CORNER:
        case Part::NORTH_EAST_CORNER:
            return cl::sycl::range<2>(halo_radius, halo_radius);
        case Part::NORTH_BORDER:
        case Part::SOUTH_BORDER:
            return cl::sycl::range<2>(width - 2 * halo_radius, halo_radius);
        case Part::WEST_BORDER:
        case Part::EAST_BORDER:
            return cl::sycl::range<2>(halo_radius, height - 2 * halo_radius);
        case Part::CORE:
            return cl::sycl::range<2>(width - 2 * halo_radius, height - 2 * halo_radius);
        default:
            throw std::invalid_argument("Invalid grid tile part specified");
        }
    }

    static cl::sycl::id<2> get_part_offset(Part part)
    {
        switch (part)
        {
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

    cl::sycl::buffer<T, 2> operator[](Part tile_part)
    {
        uindex_t part_column, part_row;
        switch (tile_part)
        {
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

        auto part_range = get_part_range(tile_part);
        uindex_t part_width = part_range[0];
        uindex_t part_height = part_range[1];

        if (!part[part_column][part_row].has_value())
        {
            auto part_range = burst_partitioned_range(part_width, part_height, burst_length);
            cl::sycl::buffer<T, 2> new_part(part_range);
            part[part_column][part_row] = new_part;
        }
        return *part[part_column][part_row];
    }

    template <cl::sycl::access::target access_target>
    void copy_from(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, access_target> accessor, cl::sycl::id<2> offset)
    {
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

    template <cl::sycl::access::target access_target>
    void copy_to(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, access_target> accessor, cl::sycl::id<2> offset)
    {
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
    template <cl::sycl::access::target access_target>
    void copy_part(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, access_target> accessor, Part part, cl::sycl::id<2> global_offset, bool buffer_to_part)
    {
        cl::sycl::id<2> offset = global_offset + get_part_offset(part);
        if (offset[0] >= accessor.get_range()[0] || offset[1] >= accessor.get_range()[1])
        {
            // Nothing to do here. There is no data in the buffer for this part.
            return;
        }

        auto part_ac = (*this)[part].template get_access<cl::sycl::access::mode::read_write>();
        auto part_range = get_part_range(part);

        uindex_t i_burst = 0;
        uindex_t i_cell = 0;
        for (uindex_t c = 0; c < part_range[0]; c++)
        {
            for (uindex_t r = 0; r < part_range[1]; r++)
            {
                if (c + offset[0] < accessor.get_range()[0] && r + offset[1] < accessor.get_range()[1])
                {
                    if (buffer_to_part)
                    {
                        part_ac[i_burst][i_cell] = accessor[c + offset[0]][r + offset[1]];
                    }
                    else
                    {
                        accessor[c + offset[0]][r + offset[1]] = part_ac[i_burst][i_cell];
                    }
                }

                if (i_cell == burst_length - 1)
                {
                    i_cell = 0;
                    i_burst++;
                }
                else
                {
                    i_cell++;
                }
            }
        }
    }

    std::optional<cl::sycl::buffer<T, 2>> part[3][3];
};
} // namespace stencil