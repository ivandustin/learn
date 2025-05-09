from pathlib import Path
from learn import checkpointer


def load(filepath: Path):
    return checkpointer.restore(filepath.resolve())
