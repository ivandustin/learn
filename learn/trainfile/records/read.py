def read(file):
    for line in file:
        line = line.strip()
        if line:
            yield list(map(float, line.split(" ")))
        else:
            yield None
    yield None
