from pathlib import Path
from jax.numpy import array
from .batch.read import read as read_batch


def read(filepath: Path):
    xs = []
    ys = []
    with open(filepath, "r") as file:
        for x, y in read_batch(file):
            xs.append(x)
            ys.append(y)
    xs = array(xs).transpose(0, 2, 1) - 1
    ys = array(ys) - 1
    return xs, ys
