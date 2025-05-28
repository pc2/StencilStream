
###  🔀 Choosing the Right Backend

The first step when using StencilStream is selecting the matching **backend** for your target hardware. Each backend is optimized for a different architecture:

1. `monotile` – FPGA (monolithic design)
2. `tiling` – FPGA (tiled design)
3. `cpu` – x86 CPUs
4. `cuda` – NVIDIA GPUs

If you're targeting a **single backend**, you can directly include the corresponding `StencilUpdate.hpp` header:

```cpp
#include <StencilStream/"backend"/StencilUpdate.hpp>
```
Replace `"backend"` with one of the supported backend names (e.g., `cpu`, `cuda`, etc.).

---

### 🔀 Backend Selection via Compiler Flags

To enable **build-time backend selection**, you can define one of the following macros during compilation:

- `STENCILSTREAM_BACKEND_MONOTILE`
- `STENCILSTREAM_BACKEND_TILING`
- `STENCILSTREAM_BACKEND_CPU`
- `STENCILSTREAM_BACKEND_CUDA`

The following snippet shows how to include the appropriate backend implementation:

```cpp
#elif defined(STENCILSTREAM_BACKEND_MONOTILE)
    #include <StencilStream/monotile/StencilUpdate.hpp>
#elif defined(STENCILSTREAM_BACKEND_TILING)
    #include <StencilStream/tiling/StencilUpdate.hpp>
#elif ...
      ...
      ...
#endif
```
