def read(file):
    for line in file:
        line = line.strip()
        if line:
            yield list(map(int, line.split(" ")))
        else:
            yield None
    yield None
