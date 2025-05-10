from learn.paths.txt import DIMS, CLASSES, NEURONS, EMBEDS
from learn.txt.list.read import read as read_list
from learn.modules.model import Model
from learn.txt.read import read
from learn import rngs


def create():
    dims = read(DIMS)
    classes = read(CLASSES)
    neurons = read(NEURONS)
    embeds = read_list(EMBEDS)
    return Model(dims, classes, embeds, neurons, rngs=rngs)
