from pathlib import Path
from learn.trainfile.read import read


def test():
    array = read(Path(__file__).parent / "train.txt")
    print(array)
