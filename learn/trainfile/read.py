from pathlib import Path
from jax.numpy import array as jax
from numpy import array, stack
from .batch.read import read as read_batch


def read(filepath: Path):
    xs = []
    ys = []
    with open(filepath, "r") as file:
        for x, y in read_batch(file):
            xs.append(array(x).T - 1)
            ys.append(array(y) - 1)
    return jax(stack(xs)), jax(stack(ys))
