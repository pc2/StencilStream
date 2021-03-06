/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "collection.hpp"
#include "defines.hpp"

// The coefficients that describe the properties of a material.
struct Material
{
    float ca;
    float cb;
    float da;
    float db;
};

// The coefficients of vaccuum.
constexpr static Material vacuum{
    (1 - (sigma * dt) / (2 * eps_0 * eps_r)) / (1 + (sigma * dt) / (2 * eps_0 * eps_r)), // ca
    (dt / (eps_0 * eps_r * dx)) / (1 + (sigma * dt) / (2 * eps_0 * eps_r)),              // cb
    (1 - (sigma * dt) / (2 * mu_0)) / (1 + (sigma * dt) / (2 * mu_0)),                   // da
    (dt / (mu_0 * dx)) / (1 + (sigma * dt) / (2 * mu_0)),                                // db
};

class FDTDKernel
{
    float disk_radius;
    float tau;
    float omega;
    float t0;
    float t_cutoff;
    uindex_t n_frame_offset;

public:
    FDTDKernel(uindex_t n_frame_offset, Parameters const &parameters) : n_frame_offset(n_frame_offset), disk_radius(parameters.disk_radius), tau(parameters.tau()), omega(parameters.omega()), t0(parameters.t0()), t_cutoff(parameters.t_cutoff()) {}

    static buffer<FDTDCell, 2> setup_cell_buffer(cl::sycl::queue queue)
    {
        buffer<FDTDCell, 2> cell_buffer(range<2>(n_buffer_columns, n_buffer_rows));

        queue.submit([&](cl::sycl::handler &cgh) {
            FDTDCell new_cell;
            new_cell.ex = float_vec(0);
            new_cell.ey = float_vec(0);
            new_cell.hz = float_vec(0);
            new_cell.hz_sum = float_vec(0);

            auto cell_buffer_ac = cell_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);

            cgh.fill(cell_buffer_ac, new_cell);
        });

        return cell_buffer;
    }

    FDTDCell operator()(
        stencil::Stencil2D<FDTDCell, stencil_radius> const &stencil,
        stencil::Stencil2DInfo const &info)
    {
        FDTDCell cell = stencil[ID(0, 0)];

        if (info.cell_generation == 0)
        {
            cell.hz_sum = 0;
        }

        float_vec center_cell_row;
#pragma unroll
        for (uindex_t i = 0; i < vector_len; i++)
        {
            center_cell_row[i] = vector_len * info.center_cell_id.r + i; // 1 + 8 FOs
        }
        float_vec center_cell_column = float_vec(info.center_cell_id.c);

        float_vec a = center_cell_row - mid_y; // 8 FOs
        float_vec b = center_cell_column - mid_x; // 8 FOs
        float_vec distance = cl::sycl::sqrt(a * a + b * b); // 8 * 4 = 32 FOs

        float_vec ca, cb, da, db;
#pragma unroll
        for (uindex_t i = 0; i < vector_len; i++)
        {
            if (distance[i] >= disk_radius)
            {
                ca[i] = da[i] = 1;
                cb[i] = db[i] = 0;
            }
            else
            {
                ca[i] = vacuum.ca;
                cb[i] = vacuum.cb;
                da[i] = vacuum.da;
                db[i] = vacuum.db;
            }
        }

        // Since info.pipeline_position is a constant, the compiler only needs to synthesize one
        // branch per core. Therefore, the mean number of FOs in this block is the mean of these branches.
        if ((info.pipeline_position & 0b1) == 0)
        {
            float_vec left_neighbours, top_neighbours;
            left_neighbours = stencil[ID(-1, 0)].hz;
            top_neighbours = stencil[ID(0, 0)].hz;
#pragma unroll
            for (uindex_t i = 0; i < vector_len - 1; i++)
            {
                top_neighbours[i + 1] = top_neighbours[i];
            }
            top_neighbours[0] = stencil[ID(0, -1)].hz[vector_len - 1];

            cell.ex *= ca; // 8 FOs
            cell.ex += cb * (stencil[ID(0, 0)].hz - top_neighbours); // 8*2 = 16 FOs

            cell.ey *= ca; // 8FOs
            cell.ey += cb * (left_neighbours - stencil[ID(0, 0)].hz); // 8*2 = 16 FOs
        }
        else
        {
            float_vec right_neighbours, bottom_neighbours;
            right_neighbours = stencil[ID(1, 0)].ey;
            bottom_neighbours = stencil[ID(0, 0)].ex;
#pragma unroll
            for (uindex_t i = 0; i < vector_len - 1; i++)
            {
                bottom_neighbours[i] = bottom_neighbours[i + 1];
            }
            bottom_neighbours[vector_len - 1] = stencil[ID(0, 1)].ex[0];

            cell.hz *= da; // 8 FOs
            cell.hz += db * (bottom_neighbours - stencil[ID(0, 0)].ex + stencil[ID(0, 0)].ey - right_neighbours); // 8*4 = 32 FOs

            uindex_t field_generation = (info.cell_generation >> 1) + n_frame_offset; // No floats, therefore no FOs
            float current_time = field_generation * dt; // 1 FO
            if (current_time < t_cutoff)
            {
                float wave_progress = (current_time - t0) / tau; // 3 FOs
                cell.hz += cl::sycl::cos(omega * current_time) * cl::sycl::exp(-1 * wave_progress * wave_progress); // 8 + 6 FOs
            }

            cell.hz_sum += cell.hz * cell.hz; // 2*8 = 16 FOs
        }
        return cell;
    }
};