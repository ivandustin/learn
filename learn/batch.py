from jax.numpy import arange
from jax.random import key, split, permutation
from .seed.create import create


def batch(size, x, y):
    master = key(create())
    total = x.shape[0]
    while True:
        master, sub = split(master)
        indices = permutation(sub, arange(total))
        for i in range(0, total, size):
            j = indices[i : i + size]
            yield x[j], y[j]
