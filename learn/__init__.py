from pathlib import Path
from time import time
from orbax.checkpoint import StandardCheckpointer
from flax.nnx import Rngs


ROOT = Path(".")
CLASSES = ROOT / "classes.txt"
FACTORS = ROOT / "factors.txt"
NEURONS = ROOT / "neurons.txt"
DIMS = ROOT / "dims.txt"
STATE = ROOT / "state"
EMBED = STATE / "embed"
MODEL = STATE / "model"
checkpointer = StandardCheckpointer()
rngs = Rngs(int(time()))
