from pathlib import Path
from jax.numpy import array
from .batch.read import read as read_batch


def read(filepath: Path):
    with open(filepath, "r") as file:
        batch = list(read_batch(file))
    xs = array([x for x, _ in batch]).transpose(0, 2, 1)
    ys = array([y for _, y in batch])
    return xs, ys
