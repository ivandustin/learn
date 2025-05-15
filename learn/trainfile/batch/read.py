from learn.trainfile.records.read import read as read_records


def read(file):
    items = []
    for record in read_records(file):
        if record is None:
            if items:
                x = items[:-1]
                last = items[-1]
                y = last[0]
                w = last[1] if len(last) > 1 else 1
                yield x, y, w
                items = []
        else:
            items.append(record)
