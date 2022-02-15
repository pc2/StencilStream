/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "GenericID.hpp"
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil {
/**
 * \brief Base class for all execution managers.
 *
 * Executors are the user-facing facades of StencilStream that orchestrate the computations.
 * Different executors may use different architectures and strategies to apply transition functions
 * to cells. Application code that may work with any executor can use this base class to access
 * them. It has multiple logical attributes that can be configured:
 *
 * ### Grid
 *
 * The grid is the logical array of cells, set with \ref AbstractExecutor.set_input. A stencil
 * executor does not work in place and a buffer used to initialize the grid can be used for other
 * tasks afterwards. The \ref AbstractExecutor.run method alters the state of the grid and the grid
 * can be copied back to a given buffer using \ref AbstractExecutor.copy_output.
 *
 * ### Transition Function
 *
 * A stencil executor stores an instance of the transition function since it may require some
 * configuration and runtime-dynamic parameters too. An instance is required for the initialization,
 * but it may be replaced at any time with \ref AbstractExecutor.set_trans_func.
 *
 * ### Generation Index
 *
 * This is the generation index of the current state of the grid. \ref AbstractExecutor.run updates
 * and therefore, it can be ignored in most instances. However, it can be reset if a transition
 * function needs it.
 *
 * \tparam T The cell type.
 * \tparam stencil_radius The radius of the stencil buffer supplied to the transition function.
 * \tparam TransFunc The type of the transition function.
 */
template <typename T, uindex_min_t stencil_radius, typename TransFunc> class AbstractExecutor {
  public:
    /**
     * \brief Create a new abstract executor.
     * \param halo_value The value of cells that are outside the grid.
     * \param trans_func The instance of the transition function that should be used to calculate
     * new generations.
     */
    AbstractExecutor(T halo_value, TransFunc trans_func)
        : halo_value(halo_value), trans_func(trans_func), i_generation(0) {}

    /**
     * \brief Compute the next generations of the grid and store it internally.
     *
     * This will use the transition function to compute the next `n_generations` generations of the
     * grid and store the new state of the grid internally. The resulting grid state can be
     * retrieved with \ref AbstractExecutor.copy_output.
     *
     * \param n_generations The number of generations to calculate.
     */
    virtual void run(uindex_t n_generations) = 0;

    /**
     * \brief Set the internal state of the grid.
     *
     * This will copy the contents of the buffer to an internal representation. The buffer may be
     * used for other purposes later. It must not reset the generation index. The range of the input
     * buffer will be used as the new grid range.
     *
     * \param input_buffer The source buffer of the new grid state.
     */
    virtual void set_input(cl::sycl::buffer<T, 2> input_buffer) = 0;

    /**
     * \brief Copy the state of the grid to a buffer.
     *
     * This will copy the cells of the internal grid representation to the buffer. The range of the
     * output buffer must be equal to the grid range (retrievable with \ref
     * AbstractExecutor.get_grid_range).
     *
     * \param output_buffer The target buffer.
     */
    virtual void copy_output(cl::sycl::buffer<T, 2> output_buffer) = 0;

    /**
     * \brief Get the range of the internal grid.
     */
    virtual UID get_grid_range() const = 0;

    /**
     * \brief Get the value of cells outside of the grid.
     */
    T const get_halo_value() const { return halo_value; }

    /**
     * \brief Set the value of cells outside of the grid.
     */
    void set_halo_value(T halo_value) { this->halo_value = halo_value; }

    /**
     * \brief Get the configured transition function instance.
     */
    TransFunc get_trans_func() const { return trans_func; }

    /**
     * \brief Set the transition function instance.
     */
    void set_trans_func(TransFunc trans_func) { this->trans_func = trans_func; }

    /**
     * \brief Get the generation index of the grid.
     */
    uindex_t get_i_generation() const { return i_generation; }

    /**
     * \brief Set the generation index of the grid.
     */
    void set_i_generation(uindex_t i_generation) { this->i_generation = i_generation; }

    /**
     * \brief Increase the generation index of the grid by a certain delta.
     */
    void inc_i_generation(index_t delta) { this->i_generation += delta; }

  private:
    T halo_value;
    TransFunc trans_func;
    uindex_t i_generation;
};
} // namespace stencil