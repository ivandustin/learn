from learn.embed.create import create as create_embed
from learn.txt.list.read import read as read_list
from learn.txt.read import read
from learn import DIMS, FACTORS


def create():
    dims = read(DIMS)
    factors = read_list(FACTORS)
    return create_embed(dims, factors)
