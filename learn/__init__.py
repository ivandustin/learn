from .seed.create import create
from flax.nnx import Rngs

rngs = Rngs(create())
