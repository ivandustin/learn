from flax.nnx import Rngs, Module, Embed
from jax.numpy import array


class Embeds(Module):
    def __init__(self, width: int, lengths: list[int], rngs: Rngs):
        self.embeds = [Embed(n, width, rngs=rngs) for n in lengths]

    def __call__(self, x):
        return array([embed(x) for embed in self.embeds])
