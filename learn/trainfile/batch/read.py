from learn.trainfile.records.read import read as read_records


def read(file):
    items = []
    for record in read_records(file):
        if record is None:
            if items:
                x = items[:-1]
                y = items[-1][0]
                yield x, y
                items = []
        else:
            items.append(record)
