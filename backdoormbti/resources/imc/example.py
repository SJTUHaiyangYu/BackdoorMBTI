# ./examples.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack imc --mark_random_init --epochs 50 --lr 0.01 --save
#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --color --verbose 1 --attack badnet --pretrained --validate_interval 1 --epochs 50 --lr 1e-2
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.attacks import BackdoorAttack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    kwargs = vars(parser.parse_args())

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BackdoorAttack = trojanvision.attacks.create(
        dataset=dataset, model=model, mark=mark, **kwargs
    )
    # print(attack.poison_set)
    if env["verbose"]:
        trojanvision.summary(
            env=env,
            dataset=dataset,
            model=model,
            mark=mark,
            trainer=trainer,
            attack=attack,
        )
    # print(attack.poison_set.data)
    # print(attack.poison_set.data[0].shape)
    # print(attack.poison_set.targets,len(attack.poison_set.targets))
    attack.attack(**trainer)
    # print(attack.poison_set)
