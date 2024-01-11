/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/Index.hpp>
#include <deque>
#include <stdexcept>

template <typename id, typename T> class HostPipe {
  public:
    static T read() { return HostPipeImpl::instance().read(); }

    static void write(T new_value) { HostPipeImpl::instance().write(new_value); }

    static bool empty() { return HostPipeImpl::instance().empty(); }

  private:
    class HostPipeImpl {
      public:
        static HostPipeImpl &instance() {
            static HostPipeImpl _instance;
            return _instance;
        }

        T read() {
            if (queue.empty()) {
                throw std::runtime_error(
                    "Try to read from empty pipe (blocking is not implemented).");
            } else {
                T new_value = queue.back();
                queue.pop_back();
                return new_value;
            }
        }

        void write(T new_value) { queue.push_front(new_value); }

        bool empty() { return queue.empty(); }

      private:
        HostPipeImpl() : queue() {}
        HostPipeImpl(const HostPipeImpl &);
        HostPipeImpl &operator=(const HostPipeImpl &);

        std::deque<T> queue;
    };
};