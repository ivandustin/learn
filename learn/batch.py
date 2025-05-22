from numpy.random import permutation


def batch(size, x, y):
    total = x.shape[0]
    while True:
        indices = permutation(total)
        for i in range(0, total, size):
            j = indices[i : i + size]
            yield x[j], y[j]
