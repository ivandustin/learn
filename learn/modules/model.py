from uat.cube import Fn
from flax.nnx import Rngs, Module
from .embeds import Embeds


class Model(Module):
    def __init__(
        self, dims: int, classes: int, embeds: list[int], neurons: int, rngs: Rngs
    ):
        self.fn = Fn(dims, classes, neurons, rngs=rngs)
        self.embeds = Embeds(dims, embeds, rngs=rngs)

    def __call__(self, x):
        return self.fn(self.embeds(x))
