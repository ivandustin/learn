from pathlib import Path
from learn.state.save import save as save_model


def save(directory: Path, models: list):
    for i, model in enumerate(models):
        filepath = directory / str(i)
        save_model(filepath, model)
    lenfile = directory / "len.txt"
    lenfile.write_text(str(len(models)))
