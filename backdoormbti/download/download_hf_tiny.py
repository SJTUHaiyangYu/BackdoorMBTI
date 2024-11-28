import json
from pathlib import Path

from datasets import load_dataset, load_from_disk

DATA_DIR = Path(__file__).parent.parent / "data"
# esc50
path = DATA_DIR / "tiny"
vggface2_train = load_dataset(
    "zh-plus/tiny-imagenet",
    split="train",
    cache_dir=path.resolve().as_posix(),
)
vggface2_test = load_dataset(
    "zh-plus/tiny-imagenet",
    split="valid",
    cache_dir=path.resolve().as_posix(),
)
# labels = sorted(esc50.unique("category"))
# with open(path / "labels.json", "w") as f:
#     json.dump(labels, f)
vggface2_train.save_to_disk(path.as_posix())
vggface2_test.save_to_disk(path.as_posix())

# test load
vggface2_train = load_from_disk(path.as_posix())
vggface2_test = load_from_disk(path.as_posix())

print("done!")
