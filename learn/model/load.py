from os.path import exists
from learn.state.load import load as fn
from .create import create


def load(checkpointer, path):
    model = create()
    if exists(path):
        model = fn(checkpointer, path, model)
    return model
