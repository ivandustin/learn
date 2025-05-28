from base64 import b64decode
from io import BytesIO
from numpy import load


def read(file):
    for line in file:
        result = load(BytesIO(b64decode(line)))
        yield result["x"], result["y"]
