from flax.nnx import Embed
from learn import rngs


def create(dims: int, factors: list[int]) -> list[Embed]:
    return [Embed(n, dims, rngs=rngs) for n in factors]
