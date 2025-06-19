## Installation

This guide walks you through setting up **StencilStream** on your system. Please note that backend availability depends on your hardware.

> ⚠️ The **framework** is supported on the [Noctua 2](https://pc2.uni-paderborn.de/de/systems-and-services/noctua-2) supercomputer but can also be used on other systems. We cannot guarantee seamless operation on other systems without adaptation.

### 🖥️ Platform Support

StencilStream is built on SYCL and supports a range of backends depending on your OS. You can target x86 CPUs, NVIDIA GPUs, and Intel FPGAs using Intel's oneAPI and Codeplay's SYCL plugins.

For each supported combination, refer to the official compatibility notes from Intel and Codeplay to ensure that your system meets the minimum software and driver requirements.

#### Linux
| Backend      | OS Requirements |
|--------------|-----------|
| x86 CPU      | • [oneAPI CPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2024.html) |
| NVIDIA GPU   | • [oneAPI NVIDIA](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2024.html) <br> • [Codeplay NVIDIA](https://developer.codeplay.com/products/oneapi/nvidia/2025.1.1/guides/get-started-guide-nvidia.html) </br> |
| Intel FPGA   | • [oneAPI FPGA](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-dpcpp/2025.html) |

---

### 📦 Software Requirements

To build and run StencilStream, you’ll need a modern SYCL toolchain and build environment. At a minimum, this includes a C++20-capable compiler, CMake, Git, and Intel's DPC++ compiler. Julia is optional but recommended for running benchmarking and visualization scripts included in the `examples/` directory.

Depending on your hardware, additional toolchains may be required, such as CUDA (for NVIDIA), or Intel’s FPGA development stack. Please ensure that all necessary device drivers and runtime libraries are correctly installed and accessible.

#### General Dependencies
| Software                     | Recommended Version      |
|-----------------------------|--------------------------|
| Intel oneAPI DPC++ Compiler | • 2024.2.1 or newer        |
| CMake                       | • 3.29.3 or newer          |
| Git                         | • 2.34 or newer                    |
| Julia (optional for benchmarking)            | • 1.10.4 or newer |

#### Backend-Specific Requirements
| OS        | Device       | Required Tools                                        |
|-----------|--------------|-------------------------------------------------------|
| Linux     | x86 CPU      | • oneAPI DPC++ Compiler                                 |
| Linux     | NVIDIA GPU   | • CUDA Toolkit <br> • oneAPI for NVIDIA GPUs by Codeplay </br>              |
| Linux     | Intel FPGA   | • Intel FPGA SDK <br> • SYCL FPGA runtime  </br>                 |


#### Product and Version Information

| Product | Supported Version |
| - | - |
| Intel oneAPI DPC++ Compiler | 2024.2.1 or newer |
| CMake | 3.29.3 or newer |
| Git | 2.34+ |
| Julia (optional for benchmarking) | 1.10.4 or newer |
| NVIDIA CUDA SDK | 12.6.0 |
| oneAPI for NVIDIA® GPUs | 2025.0.0 or newer |



> 📌 Tip: If you're targeting GPUs with Codeplay’s SYCL plugins, make sure to follow the installation instructions provided on the [Codeplay developer portal](https://developer.codeplay.com/) for your backend and platform.

----

### 🔧 Get the framework

Once your system is prepared, you can obtain the StencilStream source code by cloning the official Git repository. This will give you access to the core framework, example applications, and benchmark utilities.

Simply run the following commands to get started:

```bash
git clone https://github.com/pc2/stencilstream.git
cd stencilstream
```

From here, you can proceed with the build instructions, configure your SYCL backend, and start running examples.