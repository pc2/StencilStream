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
#include "GenericID.hpp"
#include "IOKernel.hpp"
#include "Tile.hpp"
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/queue.hpp>
#include <memory>
#include <vector>

namespace stencil
{

template <typename T, uindex_t tile_width, uindex_t tile_height, uindex_t halo_radius, uindex_t burst_length>
class Grid
{
public:
    using Tile = Tile<T, tile_width, tile_height, halo_radius, burst_length>;
    static constexpr uindex_t core_height = tile_height - 2 * halo_radius;
    static constexpr uindex_t core_width = tile_width - 2 * halo_radius;

    Grid(uindex_t width, uindex_t height) : tiles(), grid_range(width, height)
    {
        allocate_tiles();
    }

    Grid(cl::sycl::buffer<T, 2> in_buffer) : tiles(), grid_range(in_buffer.get_range())
    {
        copy_from(in_buffer);
    }

    void copy_to(cl::sycl::buffer<T, 2> &out_buffer)
    {
        if (out_buffer.get_range() != grid_range)
        {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        auto out_buffer_ac = out_buffer.template get_access<cl::sycl::access::mode::read_write>();
        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++)
        {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++)
            {
                cl::sycl::id<2> offset((tile_column - 1) * tile_width, (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_to(out_buffer_ac, offset);
            }
        }
    }

    Grid make_output_grid() const
    {
        Grid output_grid(grid_range[0], grid_range[1]);

        return output_grid;
    }

    UID get_tile_range() const
    {
        return UID(tiles.size() - 2, tiles[0].size() - 2);
    }

    UID get_grid_range() const
    {
        return grid_range;
    }

    Tile &get_tile(UID tile_id)
    {
        if (tile_id.c > get_tile_range().c || tile_id.r > get_tile_range().r)
        {
            throw std::out_of_range("Tile index out of range");
        }

        uindex_t tile_c = tile_id.c + 1;
        uindex_t tile_r = tile_id.r + 1;

        return tiles[tile_c][tile_r];
    }

    template <typename in_pipe>
    void submit_tile_input(cl::sycl::queue fpga_queue, UID tile_id)
    {
        if (tile_id.c > get_tile_range().c || tile_id.r > get_tile_range().r)
        {
            throw std::out_of_range("Tile index out of range");
        }

        uindex_t tile_c = tile_id.c + 1;
        uindex_t tile_r = tile_id.r + 1;

        submit_input_kernel<in_pipe>(fpga_queue,
                                     std::array<cl::sycl::buffer<T, 2>, 5>{
                                         tiles[tile_c - 1][tile_r - 1][Tile::Part::SOUTH_EAST_CORNER],
                                         tiles[tile_c - 1][tile_r][Tile::Part::NORTH_EAST_CORNER],
                                         tiles[tile_c - 1][tile_r][Tile::Part::EAST_BORDER],
                                         tiles[tile_c - 1][tile_r][Tile::Part::SOUTH_EAST_CORNER],
                                         tiles[tile_c - 1][tile_r + 1][Tile::Part::NORTH_EAST_CORNER],
                                     },
                                     halo_radius);

        submit_input_kernel<in_pipe>(fpga_queue,
                                     std::array<cl::sycl::buffer<T, 2>, 5>{
                                         tiles[tile_c][tile_r - 1][Tile::Part::SOUTH_WEST_CORNER],
                                         tiles[tile_c][tile_r][Tile::Part::NORTH_WEST_CORNER],
                                         tiles[tile_c][tile_r][Tile::Part::WEST_BORDER],
                                         tiles[tile_c][tile_r][Tile::Part::SOUTH_WEST_CORNER],
                                         tiles[tile_c][tile_r + 1][Tile::Part::NORTH_WEST_CORNER],
                                     },
                                     halo_radius);

        submit_input_kernel<in_pipe>(fpga_queue,
                                     std::array<cl::sycl::buffer<T, 2>, 5>{
                                         tiles[tile_c][tile_r - 1][Tile::Part::SOUTH_BORDER],
                                         tiles[tile_c][tile_r][Tile::Part::NORTH_BORDER],
                                         tiles[tile_c][tile_r][Tile::Part::CORE],
                                         tiles[tile_c][tile_r][Tile::Part::SOUTH_BORDER],
                                         tiles[tile_c][tile_r + 1][Tile::Part::NORTH_BORDER],
                                     },
                                     core_width);

        submit_input_kernel<in_pipe>(fpga_queue,
                                     std::array<cl::sycl::buffer<T, 2>, 5>{
                                         tiles[tile_c][tile_r - 1][Tile::Part::SOUTH_EAST_CORNER],
                                         tiles[tile_c][tile_r][Tile::Part::NORTH_EAST_CORNER],
                                         tiles[tile_c][tile_r][Tile::Part::EAST_BORDER],
                                         tiles[tile_c][tile_r][Tile::Part::SOUTH_EAST_CORNER],
                                         tiles[tile_c][tile_r + 1][Tile::Part::NORTH_EAST_CORNER],
                                     },
                                     halo_radius);

        submit_input_kernel<in_pipe>(fpga_queue,
                                     std::array<cl::sycl::buffer<T, 2>, 5>{
                                         tiles[tile_c + 1][tile_r - 1][Tile::Part::SOUTH_WEST_CORNER],
                                         tiles[tile_c + 1][tile_r][Tile::Part::NORTH_WEST_CORNER],
                                         tiles[tile_c + 1][tile_r][Tile::Part::WEST_BORDER],
                                         tiles[tile_c + 1][tile_r][Tile::Part::SOUTH_WEST_CORNER],
                                         tiles[tile_c + 1][tile_r + 1][Tile::Part::NORTH_WEST_CORNER],
                                     },
                                     halo_radius);
    }

    template <typename out_pipe>
    void submit_tile_output(cl::sycl::queue fpga_queue, UID tile_id)
    {
        if (tile_id.c > get_tile_range().c || tile_id.r > get_tile_range().r)
        {
            throw std::out_of_range("Tile index out of range");
        }

        uindex_t tile_c = tile_id.c + 1;
        uindex_t tile_r = tile_id.r + 1;

        submit_output_kernel<out_pipe>(fpga_queue,
                                       std::array<cl::sycl::buffer<T, 2>, 3>{
                                           tiles[tile_c][tile_r][Tile::Part::NORTH_WEST_CORNER],
                                           tiles[tile_c][tile_r][Tile::Part::WEST_BORDER],
                                           tiles[tile_c][tile_r][Tile::Part::SOUTH_WEST_CORNER],
                                       },
                                       halo_radius);

        submit_output_kernel<out_pipe>(fpga_queue,
                                       std::array<cl::sycl::buffer<T, 2>, 3>{
                                           tiles[tile_c][tile_r][Tile::Part::NORTH_BORDER],
                                           tiles[tile_c][tile_r][Tile::Part::CORE],
                                           tiles[tile_c][tile_r][Tile::Part::SOUTH_BORDER],
                                       },
                                       core_width);

        submit_output_kernel<out_pipe>(fpga_queue,
                                       std::array<cl::sycl::buffer<T, 2>, 3>{
                                           tiles[tile_c][tile_r][Tile::Part::NORTH_EAST_CORNER],
                                           tiles[tile_c][tile_r][Tile::Part::EAST_BORDER],
                                           tiles[tile_c][tile_r][Tile::Part::SOUTH_EAST_CORNER],
                                       },
                                       halo_radius);
    }

private:
    template <typename pipe>
    void submit_input_kernel(cl::sycl::queue fpga_queue, std::array<cl::sycl::buffer<T, 2>, 5> buffer, uindex_t buffer_width)
    {
        using InputKernel = IOKernel<
            T,
            halo_radius,
            core_height,
            burst_length,
            pipe,
            2,
            cl::sycl::access::mode::read>;

        fpga_queue.submit([&](cl::sycl::handler &cgh)
                          {
                              std::array<typename InputKernel::Accessor, 5> accessor{
                                  buffer[0].template get_access<cl::sycl::access::mode::read>(cgh),
                                  buffer[1].template get_access<cl::sycl::access::mode::read>(cgh),
                                  buffer[2].template get_access<cl::sycl::access::mode::read>(cgh),
                                  buffer[3].template get_access<cl::sycl::access::mode::read>(cgh),
                                  buffer[4].template get_access<cl::sycl::access::mode::read>(cgh),
                              };

                              cgh.single_task<class InputKernelLambda>([=]()
                                                                       { InputKernel(accessor, buffer_width).read(); });
                          });
    }

    template <typename pipe>
    void submit_output_kernel(cl::sycl::queue fpga_queue, std::array<cl::sycl::buffer<T, 2>, 3> buffer, uindex_t buffer_width)
    {
        using OutputKernel = IOKernel<T,
                                      halo_radius,
                                      core_height,
                                      burst_length,
                                      pipe,
                                      1,
                                      cl::sycl::access::mode::discard_write>;

        fpga_queue.submit([&](cl::sycl::handler &cgh)
                          {
                              std::array<typename OutputKernel::Accessor, 3> accessor{
                                  buffer[0].template get_access<cl::sycl::access::mode::discard_write>(cgh),
                                  buffer[1].template get_access<cl::sycl::access::mode::discard_write>(cgh),
                                  buffer[2].template get_access<cl::sycl::access::mode::discard_write>(cgh),
                              };

                              cgh.single_task<class OutputKernelLambda>([=]()
                                                                        { OutputKernel(accessor, buffer_width).write(); });
                          });
    }

    void copy_from(cl::sycl::buffer<T, 2> in_buffer)
    {
        if (in_buffer.get_range() != grid_range)
        {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        allocate_tiles();

        auto in_buffer_ac = in_buffer.template get_access<cl::sycl::access::mode::read_write>();
        for (uindex_t tile_column = 1; tile_column < tiles.size() - 1; tile_column++)
        {
            for (uindex_t tile_row = 1; tile_row < tiles[tile_column].size() - 1; tile_row++)
            {
                cl::sycl::id<2> offset((tile_column - 1) * tile_width, (tile_row - 1) * tile_height);
                tiles[tile_column][tile_row].copy_from(in_buffer_ac, offset);
            }
        }
    }

    void allocate_tiles()
    {
        tiles.clear();

        uindex_t n_tile_columns = grid_range[0] / tile_width;
        if (grid_range[0] % tile_width != 0)
        {
            n_tile_columns++;
        }
        uindex_t n_tile_rows = grid_range[1] / tile_height;
        if (grid_range[1] % tile_height != 0)
        {
            n_tile_rows++;
        }

        tiles.reserve(n_tile_columns + 2);
        for (uindex_t i_column = 0; i_column < n_tile_columns + 2; i_column++)
        {
            std::vector<Tile> column;
            column.reserve(n_tile_rows + 2);
            for (uindex_t i_row = 0; i_row < n_tile_rows + 2; i_row++)
            {
                column.push_back(Tile());
            }
            tiles.push_back(column);
        }
    }

    std::vector<std::vector<Tile>> tiles;
    cl::sycl::range<2> grid_range;
    T default_value;
};

} // namespace stencil