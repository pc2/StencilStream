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

struct FDTDCell
{
    float_vec ex, ey, hz, hz_sum;
};

class FDTDKernel
{
    double disk_radius;
    float tau;
    float omega;
    float t0;
    float t_cutoff;
    uindex_t n_time_steps;
    uindex_t n_sample_steps;

public:
    FDTDKernel(Parameters const &parameters) : disk_radius(parameters.disk_radius), tau(parameters.tau()), omega(parameters.omega()), t0(parameters.t0()), t_cutoff(parameters.t_cutoff()), n_time_steps(parameters.n_time_steps), n_sample_steps(parameters.n_sample_steps) {}

    static buffer<FDTDCell, 2> setup_cell_buffer(cl::sycl::queue queue)
    {
        buffer<FDTDCell, 2> cell_buffer(working_range);

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

        double_vec center_cell_column;
#pragma unroll
        for (uindex_t i = 0; i < vector_len; i++)
        {
            center_cell_column[i] = double(vector_len * info.center_cell_id.c + i);
        }
        double_vec center_cell_row = double_vec(info.center_cell_id.r);

        double_vec a = center_cell_row - double(mid_y);
        double_vec b = center_cell_column - double(mid_x);
        double_vec distance = cl::sycl::sqrt(a * a + b * b);

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

        if (info.may_have_sideeffects && info.cell_generation % n_sample_steps == 0 && info.cell_generation >= 0 && info.cell_generation < n_time_steps)
        {
            SampleCollectorKernel::in_pipe::write(cell.hz_sum);
            cell.hz_sum = float_vec(0);
        }

        if ((info.pipeline_position & 0b1) == 0)
        {
            float_vec left_neighbors = stencil[ID(0, 0)].hz;
#pragma unroll
            for (uindex_t i = 0; i < vector_len - 1; i++)
            {
                left_neighbors[i + 1] = left_neighbors[i];
            }
            left_neighbors[0] = stencil[ID(-1, 0)].hz[vector_len - 1];

            cell.ex *= ca;
            cell.ex += cb * (stencil[ID(0, 0)].hz - stencil[ID(0, -1)].hz);

            cell.ey *= ca;
            cell.ey += cb * (left_neighbors - stencil[ID(0, 0)].hz);
        }
        else
        {
            float_vec right_neighbors = stencil[ID(0, 0)].ey;
#pragma unroll
            for (uindex_t i = 0; i < vector_len - 1; i++)
            {
                right_neighbors[i] = right_neighbors[i + 1];
            }
            right_neighbors[vector_len - 1] = stencil[ID(1, 0)].ey[0];

            cell.hz *= da;
            cell.hz += db * (stencil[ID(0, 1)].ex - stencil[ID(0, 0)].ex + stencil[ID(0, 0)].ey - right_neighbors);

            uindex_t field_generation = info.cell_generation >> 1;
            float current_time = field_generation * dt;
            if (current_time < t_cutoff)
            {
                float wave_progress = (current_time - t0) / tau;
                cell.hz += float_vec(cl::sycl::cos(omega * current_time) * cl::sycl::exp(-1 * wave_progress * wave_progress));
            }

            cell.hz_sum += cell.hz * cell.hz;
        }
        return cell;
    }
};