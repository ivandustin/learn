from pathlib import Path
from jax.numpy import array
from .batch.read import read as read_batch


def read(filepath: Path):
    xs = []
    ys = []
    ws = []
    with open(filepath, "r") as file:
        for x, y, w in read_batch(file):
            xs.append(x)
            ys.append(y)
            ws.append(w)
    xs = array(xs).astype(int).transpose(0, 2, 1) - 1
    ys = array(ys).astype(int) - 1
    ws = array(ws).reshape(-1, 1)
    return xs, ys, ws
