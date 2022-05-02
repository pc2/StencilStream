
## Derivation of the distance measuring system:

Multiple parts of the transition function need to check whether a cell is within a certain radius of the cavity center. Let $`c`$ and $`x`$ be the column indices of the cell and the cavity center, $`r`$ and $`y`$ be the row index of the cell and the cavity center, $`r`$ be the checked radius of the disk in meters and $`dx`$ be the width or height of a cell in meters. Then, we have:

``` math
    \text{The cell is within the distance $r$ to the center} \Leftrightarrow dx \cdot \sqrt{(c - x)^2 + (r - y)^2} \leq r
```

The goal now is to to reduce the number of computations that are required to check whether this inequality is true for $`c`$ and $`r`$ with fixed $`x`$, $`y`$, and $`r`$. The values handled by the FPGA should also be integers since the column and row indices are given as integers and integer-to-float conversions are expensive. We have:

``` math
\begin{align*}
    \text{The cell is within the distance $r$ to the center} 
    &\Leftrightarrow dx \cdot \sqrt{(c - x)^2 + (r - y)^2} \leq r \\
    &\Leftrightarrow (c - x)^2 + (r - y)^2 \leq \left(\frac{r}{dx}\right)^2 \\
    &\Leftrightarrow c^2 - cx + x^2 + r^2 - ry + y^2 \leq \left(\frac{r}{dx}\right)^2 \\
    &\Leftrightarrow c(c - x) + r(r - y) \leq \left(\frac{r}{dx}\right)^2 - x^2 - y^2
\end{align*}
```

Computing the left side only requires five operations and all intermediate values on the left side and the resulting value of the right side are integers. However, they may be negative, so a signed integer type needs to be used.
