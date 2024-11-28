from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent.parent / "data"
LOG_DIR = Path(__file__).parent.parent / "logs"
TEST_DIR = Path(__file__).parent.parent / "tests"
TEST_DATA_DIR = TEST_DIR / "data"
POISON_DATA_DIR = DATA_DIR / "poison_data"

TYPES = ["image", "text", "audio"]  # , "video"]
TRAINING_CONTRAL_LIST = ["tuap"]
ATTACKS = {
    "image": [
        "badnet",
        "blend",
        "bpp",
        "wanet",
        "ssba",
        "refool",
        "sig",
        "ssba",
        "ubw",
        "embtrojan",
    ],
    "text": ["badnet", "addsent", "stylebkd", "lwp"],  # synbkd
    "audio": ["badnet", "blend", "gis", "ultrasonic", "daba", "baasv"],
    "video": ["badnet", "tuap"],
}
TARGETS = {
    "gtzan": "jazz",
    "speechcommands": "yes",
    "voxceleb1idenfication": 0,
    "esc50": "dog",
    "librispeech": "374",
    "superb": 0,
    "common_language": 25,
}
DATASETS = {
    "image": [
        "cifar10",
        # passed test
        # "cifar100",
        "gtsrb",
        "celeba",
        # "dtd",
        "tiny",
        # "imagenet"
    ],
    "text": [
        "imdb",
        "dbpedia",
        "sst2",
        "ag_news",
        # "cola",
        # "yelp",
    ],
    "audio": [
        "speechcommands",
        # passed test
        "gtzan",
        # "esc50",
        # "librispeech",
        # "superb",
        # "common_language",
        "voxceleb1idenfication",
        # not passed test
        # "musdb_hq",
        # "yesno",
        # "dr_vctk",
        # "snips",
        # "iemocap",
        # "libritts",
        "timit",
    ],
    "video": [
        "hmdb51",
        # "kinetics"
    ],
}

MODELS = {
    "image": [
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "alexnet",
        # "vgg11",
        # "vgg16",
        # "vgg19",
        # "densenet121",
        # "densenet161",
        # "mobilenet_v2",
        # "inception_v3",
        # "googlenet",
        # "shufflenet_v2_x1_0",
        # "efficientnet_b0",
        "vit_b_16",
    ],
    "text": ["bert", "gpt2", "roberta"],
    "audio": ["audiocnn", "lstm", "vggvox", "xvector"],
    "video": ["r3d"],
}
DEFENSES = [
    "ac",
    "strip",
    "finetune",
    "fineprune",
    "nc",
    "abl",
    "clp",
    "onion",
    "rap",
    "bki",
    "anp",
    "ss",
    "scale_up",
    "tabor",
    "nad",
    "mntd",
    "freeeagle",
]
POISON_RATE_LST = [
    0.000,
    0.001,
    0.005,
    0.01,
    0.04,
    0.08,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]
