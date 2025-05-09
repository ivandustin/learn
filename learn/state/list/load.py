from pathlib import Path
from learn.txt.read import read
from learn.state.load import load as load_model


def load(directory: Path):
    lenfile = directory / "len.txt"
    length = read(lenfile)
    return [load_model(directory / str(i)) for i in range(length)]
