#!/usr/bin/env python3
from matplotlib import pyplot
from sys import argv
import numpy as np

if len(argv) < 3:
    print("Usage: {} <file.csv> <file.png>".format(argv[0]))
    exit()

values = np.asarray([float(line) for line in open(argv[1], "r")])

width = height = 4096
assert(len(values) == width * height)

values = values.reshape((width, height), order='C')

pyplot.pcolormesh(values)
pyplot.savefig(argv[2], format="png")