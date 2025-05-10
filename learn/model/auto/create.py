from learn.model.create import create as create_model
from learn.paths.txt import DIMS, CLASSES, NEURONS
from learn.txt.read import read


def create():
    dims = read(DIMS)
    classes = read(CLASSES)
    neurons = read(NEURONS)
    return create_model(dims, classes, neurons)
