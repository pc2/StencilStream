#!/usr/bin/env python3
from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation
from sys import argv
import numpy as np
from pathlib import PosixPath
import re
import csv
from math import inf

if len(argv) != 2:
    print("Usage: {} <output dir>".format(argv[0]))
    exit()

output_dir = PosixPath(argv[1])
assert(output_dir.is_dir())

fig, ax = pyplot.subplots()

number_re = re.compile("([0-9]+)")
frame_files = [(int(number_re.search(path.stem)[1]), path) for path in output_dir.glob("*.csv")]
frame_files.sort()

frames = []
min_value, max_value = inf, -inf
for _, path in frame_files:
    with open(path) as frame:
        frame = np.asarray([[float(cell) for cell in row] for row in csv.reader(frame)], dtype=np.float32)
    frames.append(frame)
    min_value = min(min_value, frame.min())
    max_value = max(max_value, frame.max())

frames = [[ax.imshow(frame, animated=True, origin="lower", vmin=min_value, vmax=max_value)] for frame in frames]

ax.set_title("Simulated convection")

ani = ArtistAnimation(fig, frames, blit=True, interval=50)
ani.save("animation.mp4", fps=24)
