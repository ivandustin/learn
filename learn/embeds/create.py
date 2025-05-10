from functools import partial
from learn.modules.embeds import Embeds
from learn import rngs

create = partial(Embeds, rngs=rngs)
