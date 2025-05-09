from uat.cube import Fn
from learn import rngs


def create(dims: int, classes: int, neurons: int):
    return Fn(dims, classes, neurons, rngs=rngs)
