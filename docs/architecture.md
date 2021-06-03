# Architecture

### Terminology {#terms}

First, we have to lay down some terminology used in StencilStream:

| Word | Definition |
|------|------------|
| static | defined at compile time, part of the design |
| dynamic | defined at runtime, part of the payload data |
| Cell | Fundamental element of StencilStream's architecture. Its type is user-provided and every cell has a value of this type. |
| Grid | A rectangular container of cells of a dynamic, arbitrary size |
| Grid width/height | The dynamic number of columns/rows in a grid |
| Stencil | A rectangular container with a central cell and all cells at a [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) up to the stencil radius ([extended Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)) |
| Stencil radius | The static, maximal Chebyshev distance of cells in a stencil to the central cell |
| Stencil diameter | `2 * stencil_radius + 1`, the width and height of the stencil |
| Tile | A rectangular container of cells of a static size |
| Tile width/height | The number of columns/rows in a tile |
| Transitition function | A function that maps a stencil onto the next generation of the stencil's central cell |

### Indexing and Iteration order {#index}

Rectangular containers in StencilStream are always organized in columns and rows. The first index is always the column index and the second index is always the row index. The origin is thought to be in the north-western corner. An example where every grid cell contains it's index:

```
                       North
     -----------------------------------------
     | (0,0) | (1,0) | (2,0) | (3,0) | (4,0) |
     -----------------------------------------
     | (0,1) | (1,1) | (2,1) | (3,1) | (4,1) |
     -----------------------------------------
West | (0,2) | (1,2) | (2,2) | (3,2) | (4,2) | East
     -----------------------------------------
     | (0,3) | (1,3) | (2,3) | (3,3) | (4,3) |
     -----------------------------------------
     | (0,4) | (1,4) | (2,4) | (3,4) | (4,4) |
     -----------------------------------------
                       South
```

Iteration over rectangular containers in StencilStream is always column wise, meaning that the row index is moving the fastest. An example of a for loop that iterates over a Grid with `grid_width` columns and `grid_height` rows:

```
for (uindex_t c = 0; c < grid_width; c++)
{
    for (uindex_t r = 0; r < grid_height; r++)
    {
        grid[c][r] = ID(c, r);
    }
}
```

### Architecture {#model}

The general goal of a stencil execution engine like StencilStream is to calculate a certain generation of a grid of cells. This is done by looking at the neighbourhood of a cell (contained in the stencil) and calculating the next generation of this cell, based in its neighbourhood. This is done iteratively over all cells of the grid and for all generations that have to be computed. Storing all cells in a buffer and reading the neihbourhood from this buffer doesn't work well for FPGAs. StencilStream therefore uses an approach introduced by [Hamid Reza Zohouri, Artur Podobas and Satoshi Matsuoka](https://dl.acm.org/doi/pdf/10.1145/3174243.3174248) that uses a spatially tiled buffer and temporal caching to perform the computations.

#### Tile-wise computation {#tiles}

Let's look at the simplest case first: There is a tile and we want to calculate the next generation of it. StencilStream splits up this task into an input kernel, an execution kernel and an output kernel which communicate via on-chip FIFO pipes. The input kernel get access to a buffer with the current generation of the tile and the output kernel gets access to a buffer for the next generation of the tile. The input kernel reads the cells from the buffer column-wise (as discussed in ["Indexing and Iteration order"](#index)) and sends them to the execution kernel.

The execution kernel internally has a stencil buffer stored in registers and a cache stored in on-chip memory. This cache has a width of `stencil_diameter - 1` and is as high as the input buffer. When a new cell arrives from the input kernel, every cell in the stencil buffer is shifted to the north once and the new cell is placed in the south-eastern corner of the stencil buffer. The rest of the southern-most row is filled with cells from the cache. Then, the row in the cache is overriden with all the cells in southern-most row of the stencil buffer, except for the western-most cell in this row. In effect, this means that all cells in this row of the cache are shifted west once and the eastern-most cell in this row is set to the input. The following figure illustrates this:

![Shifting](shifting.svg)

After these shifts, the stencil buffer contains the correct neighbourhood of a central cell. Then, the transition function is executed to compute the next generation of the central cell. The result is sent to the output kernel which writes it to the output buffer, the row counter is increased and the next input is read.

Of course, the input and output of this execution stage do not necessarily have to come from the input or output kernel. StencilStream arranges multiple execution stages into a pipeline. This means that for a given pipeline length of `p`, the grid is only written to global memory every `p` generations and since the main loop of the execution kernel is pipelined itself, all of these `p` generations are calculated in parallel, utilizing the full potential of the FPGA.

#### Grid Tiling {#tiling}

The execution kernel described above works on tiles, which have a static size, but the user provides a grid, which has a dynamic size. Therefore, the host needs to partition the grid into tiles. This becomes problematic at the borders of a tile: In the grid, a cell at the border of a tile might have a neighbour which is needed to calculate the next generation. However, this neighbour is not contained in the tile and therefore, the input has to contain more cells from neighbouring tiles. In fact, for every stage in the pipeline, the input needs `r` more cells in every cardinal direction where `r` is the stencil radius. The host therefore partions the tiles too to make access easier.
