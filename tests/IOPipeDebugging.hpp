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
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>

class InputPipeWriter {
  public:
    InputPipeWriter(std::size_t i_pipe, std::mutex &mutex)
        : pipe_lock(mutex),
          pipe_file(std::to_string(2 * i_pipe),
                    std::ios_base::out | std::ios_base::app | std::ios_base::binary) {}

    InputPipeWriter(InputPipeWriter &) = delete;
    void operator=(InputPipeWriter &) = delete;

    template <typename T>
    void write(T value)
        requires(sizeof(T) == 32)
    {
        pipe_file.write((char *)&value, sizeof(T));
    }

  private:
    std::unique_lock<std::mutex> pipe_lock;
    std::fstream pipe_file;
};

class OutputPipeReader {
  public:
    OutputPipeReader(std::size_t i_pipe, std::mutex &mutex, std::size_t &read_bytes)
        : pipe_lock(mutex),
          pipe_file(std::to_string(2 * i_pipe + 1), std::ios_base::in | std::ios_base::binary),
          read_bytes(read_bytes) {
        pipe_file.seekg(read_bytes);
    }

    OutputPipeReader(InputPipeWriter &) = delete;
    void operator=(OutputPipeReader &) = delete;

    template <typename T>
    T read()
        requires(sizeof(T) == 32)
    {
        T value;
        pipe_file.read((char *)&value, sizeof(T));
        if (pipe_file.gcount() != 32) {
            throw std::runtime_error("Unable to read from an IO pipe");
        }
        read_bytes += 32;
        return value;
    }

  private:
    std::unique_lock<std::mutex> pipe_lock;
    std::fstream pipe_file;
    std::size_t &read_bytes;
};

class IOPipeDebugManager {
  public:
    // Taken from
    // https://stackoverflow.com/questions/1008019/how-do-you-implement-the-singleton-design-pattern
    static IOPipeDebugManager &get_instance() {
        static IOPipeDebugManager instance;
        return instance;
    }

    IOPipeDebugManager(IOPipeDebugManager const &) = delete;
    void operator=(IOPipeDebugManager const &) = delete;

    InputPipeWriter get_input_pipe_writer(std::size_t i_pipe) {
        if (i_pipe >= 4) {
            throw std::out_of_range("Pipe index out of range");
        }
        return InputPipeWriter(i_pipe, input_pipe_mutex[i_pipe]);
    }

    OutputPipeReader get_output_pipe_reader(std::size_t i_pipe) {
        if (i_pipe >= 4) {
            throw std::out_of_range("Pipe index out of range");
        }
        return OutputPipeReader(i_pipe, output_pipe_mutex[i_pipe], read_bytes[i_pipe]);
    }

  private:
    IOPipeDebugManager()
        : input_pipe_mutex{std::mutex(), std::mutex(), std::mutex(), std::mutex()},
          output_pipe_mutex{std::mutex(), std::mutex(), std::mutex(), std::mutex()},
          read_bytes{0, 0, 0, 0} {
        remove_files();
    }

    ~IOPipeDebugManager() { remove_files(); }

    void remove_files() {
        for (const char *path : {"0", "1", "2", "3", "4", "5", "6", "7"}) {
            if (std::filesystem::is_regular_file(path)) {
                std::remove(path);
            }
        }
    }

    std::mutex input_pipe_mutex[4];
    std::mutex output_pipe_mutex[4];
    std::size_t read_bytes[4];
};