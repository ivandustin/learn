from time import time
from flax.nnx import Rngs

seed = int(time())
rngs = Rngs(seed)
