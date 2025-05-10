from learn.state.load import load as load_model
from learn import MODEL
from .create import create


def load():
    return load_model(MODEL, create())
