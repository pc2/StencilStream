# What's new in StencilStream 4.0.0

StencilStream 4.0.0 broadens the framework from an FPGA-focused stencil
accelerator into a portable, SYCL-based 2D stencil framework that targets FPGAs,
NVIDIA GPUs, and CPUs from the same transition-function code. It also brings a
significant API cleanup, makes the FPGA backends faster through spatial
parallelism, and adds an experimental multi-FPGA execution path.

## New backends and performance features

* **GPU backend.** A new CUDA backend, built on
  [Codeplay's oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/home/index.html),
  brings StencilStream to NVIDIA GPUs. A transparent
  Array-of-Structs ↔ Struct-of-Arrays transformation lets the very same
  transition function reach high throughput on both GPUs and FPGAs. See the
  IWOCL '26 publication ([doi:10.1145/3811257.3811259](https://doi.org/10.1145/3811257.3811259))
  for a detailed evaluation.
* **Spatial parallelism on the FPGA backends.** The Monotile and Tiling
  backends now allow vectorizing the I/O and update kernels, raising the achievable
  throughput substantially over 3.0.0. The highest single-device throughput
  measured for the new release is 176.08 GCells/s (1.58 TFLOPS) for Jacobi on
  the Tiling backend, and 122.67 GCells/s (1.84 TFLOPS) arithmetic throughput for HotSpot on the
  Monotile backend (BittWare 520N / Intel Stratix 10 GX 2800).
* **Experimental multi-FPGA Monotile backend.** Uses the networking
  capabilities of high-end FPGAs to scale a Monotile design beyond a single
  device.

## Breaking API changes

These changes affect every user upgrading from 3.0.0:

* **Index types.** The configurable `stencil::uindex_t` and `stencil::index_t`,
  along with the `STENCIL_INDEX_WIDTH` macro, have been replaced by
  `std::size_t` and `std::ptrdiff_t` to align with the SYCL standard. Index width
  narrowing is now done automatically within the FPGA backends.
  The `StencilStream/Index.hpp` header has been removed.
* **2D coordinates.** The custom `ID` / `UID` / `GenericID` types have been
  replaced by `sycl::id<2>` and `sycl::range<2>`, again to align with the SYCL standard.
   The `StencilStream/GenericID.hpp` header has been removed.
* **Index ordering.** The first index of a 2D coordinate is now the row and the
  second is the column, again matching the SYCL standard. Transition functions,
  grid construction, and accessor calls written against 3.0.0 must be updated
  accordingly.
* **No more Boost dependency.** StencilStream no longer pulls in Boost;
  builds and downstream projects can drop the corresponding find/link lines.
* **Internal headers reorganized.** Implementation-detail headers (helpers,
  I/O / memory / switch kernels, the per-backend kernel and design classes)
  now live under `StencilStream/internal/` and per-backend `internal/`
  subdirectories. Public concepts in `Concepts.hpp` and `Stencil.hpp` have
  been updated to use the new index types.

## New example and documentation

* **Jacobi example.** A new example under `examples/jacobi/` provides multiple
  Jacobi-kernel variants with adjustable computational complexity, and serves
  as the primary benchmark in the new performance figures.
* **Documentation overhaul.** The README has been rewritten with up-to-date
  build, run, and benchmarking instructions covering all backends, and the
  Doxygen documentation now uses the Doxygen Awesome theme with a dark-mode
  toggle.

## Build and tooling

* **Toolchain.** Validated on Intel oneAPI 24.2.1.
* **Environment setup.** Separate `scripts/env_fpga.sh` and
  `scripts/env_cuda.sh` scripts replace the previous combined setup, so the
  FPGA and CUDA toolchains can be loaded independently on Noctua 2.
* **Per-backend benchmark scripts.** Each example now ships
  `benchmark_mono.sh`, `benchmark_tiling.sh`, and `benchmark_cuda.sh` driver
  scripts on top of the shared Julia benchmark harness.
* **Standalone Conway build.** The Conway example provides a
  `CMakeLists.standalone.txt` that can be used to build it outside of the
  StencilStream source tree.
