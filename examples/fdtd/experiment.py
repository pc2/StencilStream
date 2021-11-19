#!/usr/bin/env python3
# Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
====================================
Experiment building library for FDTD
====================================

This library contains classes to describe an experiment setup for the FDTD application. The typical
workflow is to execute this library as a script, which opens an iPython shell. There, you create an
Experiment object, populate it with your experiment and export it as a JSON file.
"""
from typing import List, Union
from pathlib import Path
from math import sqrt
import json

class Source(object):
    def __init__(self, x: float, y: float):
        self.x: int = x
        self.y: int = y
        self.t_start: float = 0.0
        self.t_cutoff: float = 7.0
        self.frequency: float = 120e12
        self.phase: float = 3.0

class Material(object):
    def __init__(self, relative_permeability: float, relative_permittivity: float):
        self.relative_permeability = relative_permeability
        self.relative_permittivity = relative_permittivity

    @classmethod
    def vacuum(cls) -> 'Material':
        return cls(1.0, 1.0)

    @classmethod
    def perfect_metal(cls) -> 'Material':
        return cls(float('inf'), float('inf'))

class Experiment:
    def __init__(self, width: float, height: float, dx: float, base_material: Material = Material.vacuum()):
        self.tau: float = 100e-15
        self.t_detect: float = 14.0
        self.t_max: float = 15.0
        self.dx: float = dx
        self.sources: List[Source] = [Source(width / 2, height / 2)]
        self.materials: List[Material] = [base_material]
        self.field: List[List[int]] = [[0 for _ in range(int(width / self.dx))] for _ in range(int(height / self.dx))]

    @property
    def n_columns(self) -> int:
        return len(self.field)

    @property
    def n_rows(self) -> int:
        if len(self.field) == 0:
            return 0
        else:
            return len(self.field[0])

    def draw_circle(self, x_center: float, y_center: float, radius: float, i_material: int):
        if i_material >= len(self.materials):
            raise Exception("Material index out of range")

        for c in range(self.n_columns):
            for r in range(self.n_rows):
                delta_x = c*self.dx - x_center
                delta_y = r*self.dx - y_center
                distance = sqrt(delta_x**2 + delta_y**2)
                if distance <= radius:
                    self.field[c][r] = i_material

    def draw_rectangle(self, x: float, y: float, width: float, height: float, i_material: int):
        if i_material >= len(self.materials):
            raise Exception("Material index out of range")

        for c in range(self.n_columns):
            for r in range(self.n_rows):
                if x <= c * self.dx <= x + width and y <= r * self.dx <= y + height:
                    self.field[c][r] = i_material

    def export(self, path: Union[str, Path] = "experiment.json"):
        data = self.__dict__
        data["sources"] = [s.__dict__ for s in list(data["sources"])]
        data["materials"] = [m.__dict__ for m in list(data["materials"])]
        json.dump(data, open(path, mode="w"))

if __name__ == "__main__":
    print(__doc__)
    import IPython
    IPython.embed()