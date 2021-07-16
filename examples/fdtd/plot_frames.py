#!/usr/bin/env python3
from matplotlib import pyplot
from matplotlib.colors import Normalize
from sys import argv
import numpy as np
from pathlib import PosixPath
from multiprocessing import Pool
import csv

if len(argv) != 2:
    print("Usage: {} <output dir>".format(argv[0]))
    exit()

output_dir = PosixPath(argv[1])
assert(output_dir.is_dir())

def plot_frame(path):
    with open(path) as frame:
        array = np.asarray([[float(cell) for cell in row[:-2]] for row in csv.reader(frame, skipinitialspace=True)], dtype=np.float32)
    local_max = array.max()
    pyplot.pcolormesh(array, norm=Normalize(vmin=0.0, vmax=local_max, clip=True))
    print("Scaling to 0.0 ... " + str(local_max))
    path = path.with_suffix(".png")
    pyplot.savefig(path, format="png")

with Pool() as pool:
    pool.map(plot_frame, (out_file for out_file in output_dir.glob("*.csv")))
