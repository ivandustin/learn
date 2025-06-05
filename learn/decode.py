from base64 import b64decode
from io import BytesIO
from numpy import load


def decode(batch: str):
    with load(BytesIO(b64decode(batch))) as result:
        return result["x"], result["y"]
