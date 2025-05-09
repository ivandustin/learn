from pathlib import Path
from time import time
from orbax.checkpoint import StandardCheckpointer
from flax.nnx import Rngs


ROOT = Path(".")
FACTORS = ROOT / "factors.txt"
DIMS = ROOT / "dims.txt"
STATE = ROOT / "state"
EMBED = STATE / "embed"
checkpointer = StandardCheckpointer()
rngs = Rngs(int(time()))
