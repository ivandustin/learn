from base64 import b64encode
from io import BytesIO
from numpy import savez


def encode(x, y) -> str:
    buffer = BytesIO()
    savez(buffer, x=x, y=y)
    return b64encode(buffer.getvalue()).decode("utf-8")
