from base64 import b64decode
from io import BytesIO
from numpy import load


def decode(batch: str):
    return load(BytesIO(b64decode(batch)))
