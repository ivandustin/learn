from learn.paths.txt import DIMS, CLASSES, NEURONS, EMBEDS
from learn.txt.list.read import read as read_list
from learn.modules.model import Model
from learn.txt.read import read
from learn import rngs


def create():
    embeds = read_list(EMBEDS)
    classes = read(CLASSES)
    dims = read(DIMS)
    neurons = read(NEURONS)
    return Model(embeds, classes, dims, neurons, rngs=rngs)
