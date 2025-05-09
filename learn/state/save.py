from flax.nnx import split
from learn import checkpointer


def save(filepath, model):
    _, state = split(model)
    checkpointer.save(filepath.resolve(), state)
