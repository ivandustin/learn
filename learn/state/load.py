from pathlib import Path
from flax.nnx import split, merge
from learn import checkpointer


def load(filepath: Path, model):
    graph, _ = split(model)
    state = checkpointer.restore(filepath.resolve())
    return merge(graph, state)
