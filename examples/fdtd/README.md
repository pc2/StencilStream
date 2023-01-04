## Building

The FDTD example uses CMake as its build system. It is prepared by running:

``` bash
mkdir build
cd build
cmake ..
```

If verbose synthesis outputs are desired, one can add the definition `-DFDTD_VERBOSE_SYNTHESIS=1`. There are different interchangeable components to the FDTD application, some of which can be chosen independently from each other. Each variant of FDTD has a different name of the form `fdtd_<Material Resolver>_<TDVS>_<Backend>[_<Modifier]` where the modifier part is optional. The different components of the name are described below. One does build the variant by execution `make fdtd_...` in the build directory.

The `<Material Resolver>` denotes how the material of a cell is stored in a cell and how the  material coefficients are retrieved from it. Possible values are:
* `coef`: Store the final material coefficients directly in every cell.
* `lut`: Store a lookup table with all known material coefficients in the processing elements and store only an index in the cell.
* `render`: Use a lookup table like with `lut`, but pick the material depending on the cell's position. No material information is stored in the cells.

The `<TDVS>` part denotes how and where the computations of time-dependent values, i.e. the current time and the source wave amplitude, are done. Possible values are:
* `inline`: Compute time-dependent values inside the processing elements.
* `device`: Precompute time-dependent values on the device and store them in a lookup table.
* `host`: Precompute time-dependent values on the host and store them in a lookup table.

StencilStream offers different backends or executors with different architectures or goals; The `<Backend>` part denotes this backend. The possible values are:
* `mono`: Use the monotile FPGA backend of StencilStream. It yields a higher performance for the same number of processing elements than `tiling`, but it is limited to a maximal grid width and height.
* `tiling`: Use the tiling FPGA backend of StencilStream. It can handle arbitrarily large grids (and therefore cavity radii and resolutions), but generally yields a lower performance than `mono`.
* `cpu`: Use the testing CPU backend of StencilStream. This backend is a trivial implementation of the executor interface for CPUs and therefore performs worse than both FPGA backends when synthesized, but it's good enough to complete most simulations in reasonable times for testing purposes.

The FPGA backends also support some modifiers which are denoted in the optional `<Modifier>` part.
* `emu`: Don't synthesize the design and create an emulation image. Note that emulation is *very*  slow and not suitable to test the functionality of the transition function. Use the `cpu` backend instead.
* `report`: Generate a hardware usage report for the variant. These reports can deliver a good estimate of the synthesized design's performance.

All of these options are interchangeable. For example, `fdtd_render_inline_mono` is a compiled FPGA design that uses the monotile architecture, computes the cell's material from its position, and computes time-dependent values inside the processing elements. `fdtd_render_host_mono` on the other hand is mostly the same, but it precomputes the time-dependent values on the host. There is obviously a large number of variants, but it is possible to build all CPU-only executables, FPGA emulation executables, hardware reports, and even all FPGA hardware executables via the meta-targets `cpus`, `emus`, `reports`, and `hws`.

## Derivation of the distance measuring system:

Multiple parts of the transition function need to check whether a cell is within a certain radius of the cavity center. Let $`c`$ and $`x`$ be the column indices of the cell and the cavity center, $`r`$ and $`y`$ be the row indices of the cell and the cavity center, $`ra`$ be the checked radius of the disk in meters and $`dx`$ be the width or height of a cell in meters. Then, we have:

``` math
    \text{The cell is within the distance $r$ to the center} \Leftrightarrow dx \cdot \sqrt{(c - x)^2 + (r - y)^2} \leq ra
```

The goal now is to to reduce the number of computations that are required to check whether this inequality is true for $`c`$ and $`r`$ with fixed $`x`$, $`y`$, and $`r`$. The values handled by the FPGA should also be integers since the column and row indices are given as integers and integer-to-float conversions are expensive. We have:

``` math
\begin{align*}
    \text{The cell is within the distance $r$ to the center} 
    &\Leftrightarrow dx \cdot \sqrt{(c - x)^2 + (r - y)^2} \leq ra \\
    &\Leftrightarrow (c - x)^2 + (r - y)^2 \leq \left(\frac{ra}{dx}\right)^2 \\
    &\Leftrightarrow c^2 - 2cx + x^2 + r^2 - 2ry + y^2 \leq \left(\frac{ra}{dx}\right)^2 \\
    &\Leftrightarrow c(c - 2x) + r(r - 2y) \leq \left(\frac{ra}{dx}\right)^2 - x^2 - y^2
\end{align*}
```

Computing the left side only requires five operations and all intermediate values on the left side and the resulting value of the right side are integers. However, they may be negative, so a signed integer type needs to be used.
