from .model.load import load as fn
from .paths.state import MODEL


def load(checkpointer):
    return fn(checkpointer, MODEL)
