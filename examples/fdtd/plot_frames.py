#!/usr/bin/env python3
from matplotlib import pyplot
from matplotlib.colors import Normalize
from sys import argv
import numpy as np
from pathlib import PosixPath

if len(argv) < 4:
    print("Usage: {} <output dir> <width> <height>".format(argv[0]))
    exit()

output_dir = PosixPath(argv[1])
assert(output_dir.is_dir())

width = int(argv[2])
height = int(argv[3])

values = dict()
max_value = 0.0

for out_file in output_dir.glob("*.csv"):
    new_array = np.asarray([float(line) for line in open(out_file, "r")])
    assert(len(new_array) == (width * height))
    max_value = max(max_value, max(new_array))
    values[out_file] = new_array.reshape((width, height), order='C')

for (path, array) in values.items():
    pyplot.pcolormesh(array, norm=Normalize(vmin=0.0, vmax=max_value, clip=True))
    path = path.with_suffix(".png")
    pyplot.savefig(path, format="png")