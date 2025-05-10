from learn.embeds.create import create as create_embeds
from learn.txt.list.read import read as read_list
from learn.paths.txt import DIMS, EMBEDS
from learn.txt.read import read


def create():
    return create_embeds(read(DIMS), read_list(EMBEDS))
