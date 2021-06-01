# Architecture

### Terminology

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

### Indexing and Iteration order 

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

Iteration over Rectangular containers in StencilStream is always column wise, meaning that the row index is moving the fastest. An example of a for loop that iterates over a Grid with `grid_width` columns and `grid_height` rows:

```
for (uindex_t c = 0; c < grid_width; c++)
{
    for (uindex_t r = 0; r < grid_height; r++)
    {
        grid[c][r] = ID(c, r);
    }
}
```