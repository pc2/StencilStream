#!/usr/bin/env python3
from matplotlib import pyplot
from matplotlib.colors import Normalize
from sys import argv
import numpy as np
from pathlib import PosixPath
from multiprocessing import Pool

if len(argv) < 4:
    print("Usage: {} <output dir> <width> <height>".format(argv[0]))
    exit()

output_dir = PosixPath(argv[1])
assert(output_dir.is_dir())

width = int(argv[2])
height = int(argv[3])

def get_max_value(path):
    return max(float(line) for line in open(path, "r"))

def plot_frame(path):
    local_max = get_max_value(path)
    array = np.asarray([float(line) for line in open(path, "r")]).reshape((width, height), order='C')
    pyplot.pcolormesh(array, norm=Normalize(vmin=0.0, vmax=local_max, clip=True))
    print("Scaling to 0.0 ... " + str(local_max))
    path = path.with_suffix(".png")
    pyplot.savefig(path, format="png")

with Pool() as pool:
    pool.map(plot_frame, (out_file for out_file in output_dir.glob("*.csv")))
