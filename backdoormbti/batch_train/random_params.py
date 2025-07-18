import random
import time


def benign_random_params(args):
    random.seed(time.time())


def badnet_random_params(args):
    random.seed(time.time())
    args.cx = random.randint(0, 28)
    args.cy = random.randint(0, 28)
    args.pratio = round(random.uniform(0, 0.5), 2)
    print(f"axis: {args.cx, args.cy}; pratio:{args.pratio}")


def bpp_random_params(args):
    # s k input_height
    random.seed(time.time())
    args.random_crop = random.randint(0, 10)
    args.random_rotation = random.randint(0, 180)
    args.squeeze_num = random.randint(2, 64)
    args.dithering = random.choice([True, False])
    args.pratio = round(random.uniform(0, 0.5), 2)
    print(
        f"random_crop :{args.random_crop} ; random_rotation:{args.random_rotation};squeeze_num: {args.squeeze_num};dithering:{args.dithering};pratio:{args.pratio}"
    )


def sig_random_params(args):
    # s k input_height
    random.seed(time.time())
    strings = ["sin", "ramp", "triangle"]
    # random select a char
    args.poisonType = random.choice(strings)
    args.pratio = round(random.uniform(0, 0.5), 2)
    print(f"poisonType:{args.poisonType}; pratio:{args.pratio}")


def want_random_params(args):
    # s k input_height
    random.seed(time.time())
    args.s = random.random()
    args.k = random.randint(1, args.input_height)
    args.pratio = round(random.uniform(0, 0.5), 2)
    print(f"s :{args.s} ; k:{args.k};pratio: {args.pratio}")
