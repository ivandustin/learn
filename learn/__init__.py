from time import time
from orbax.checkpoint import StandardCheckpointer
from flax.nnx import Rngs

checkpointer = StandardCheckpointer()
rngs = Rngs(int(time()))
