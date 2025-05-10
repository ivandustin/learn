from learn.state.load import load as load_state
from learn.embeds.auto.create import create
from learn.paths.state import EMBEDS


def load():
    return load_state(EMBEDS, create())
