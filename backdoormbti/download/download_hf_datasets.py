import json
from pathlib import Path

from datasets import load_dataset, load_from_disk

DATA_DIR = Path(__file__).parent.parent / "data"
# esc50
path = DATA_DIR / "ESC50"
esc50 = load_dataset("ashraq/esc50", split="train", cache_dir=path.resolve().as_posix())
labels = sorted(esc50.unique("category"))
with open(path / "labels.json", "w") as f:
    json.dump(labels, f)
esc50.save_to_disk(path.as_posix())

# test load
esc50 = load_from_disk(path.as_posix())
# common_language
path = DATA_DIR / "CommonLanguage"
cl_train = load_dataset(
    "common_language", split="train", cache_dir=path.resolve().as_posix()
)
labels = sorted(cl_train.unique("language"))
with open(path / "labels.json", "w") as f:
    json.dump(labels, f)
train_path = path / "train"
cl_train.save_to_disk(train_path)
cl_test = load_dataset(
    "common_language", split="test", cache_dir=path.resolve().as_posix()
)
test_path = path / "test"
cl_test.save_to_disk(test_path)

# test load
cl_train = load_from_disk(train_path.as_posix())
cl_test = load_from_disk(test_path.as_posix())

# superb
path = DATA_DIR / "Superb"
superb_train = load_dataset(
    "superb", "ks", split="train", cache_dir=path.resolve().as_posix()
)
labels = sorted(superb_train.unique("label"))
with open(path / "labels.json", "w") as f:
    json.dump(labels, f)
train_path = path / "train"
superb_train.save_to_disk(train_path.as_posix())
superb_test = load_dataset(
    "superb", "ks", split="test", cache_dir=path.resolve().as_posix()
)
test_path = path / "test"
superb_test.save_to_disk(test_path.as_posix())

# test load
spb_train = load_from_disk(train_path.as_posix())
spb_test = load_from_disk(test_path.as_posix())


print("done!")
