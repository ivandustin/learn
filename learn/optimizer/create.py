from flax.nnx import Optimizer
from optax import adam


def create(model):
    return Optimizer(model, adam(1e-3))
