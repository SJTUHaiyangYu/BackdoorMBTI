# Input-Aware Dynamic Backdoor Attack

This is an implematation of the NeurIPS 2020 paper **[Input-Aware Dynamic Backdoor Attack](https://papers.nips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)** in PyTorch.

```
@article{nguyen2020inputaware,
  title={Input-Aware Dynamic Backdoor Attack},
  author={Nguyen, Anh and Tran, Anh},
  booktitle={Proceedings of Advances in Neural Information Processing Systems},
  year={2020}
}
```


## Training code
Run command
```bash
$ python train.py --dataset <datasetName> --attack_mode <attackMode>
```
where the parameters are the following:
- `<datasetName>`: `mnist`|`cifar10`|`gtsrb`
- `<attackMode>`: `all2one` (single-target attack) or `all2all` (multi-target attack)

The trained model with be saved at the path `checkpoints/<datasetName>/<attackMode>/<attackMode>_<datasetName>_ckpt.pth.tar`

## Pretrained models
You can download the pretrained `all2one` models used in our paper [here](https://drive.google.com/file/d/1b8-1Susq5vk_QXadLS9pH45F68z3FAOS/view?usp=sharing). Just extract the zip file inside the `checkpoints` folder to use these models.

## Evaluation code
Run command
```bash
$ python eval.py --dataset <datasetName> --attack_mode <attackMode>
```
This command will print the model accuracies on three tests: clean, attack, and cross-trigger test. The clean and attack accuracies should be the same as reported in our paper, while cross-trigger one may be slightly different due to random pairing.
Besides, our code will also generate three sample sets of backdoor images named `<datasetName>_<attackMode>_sample_<index>.png` with `<index>` as `0`, `1`, and `2`.

## Results
| Dataset         | Clean test  | Attack test | Cross-trigger test |
|-----------------|-------------|-------------|--------------------|
| MNIST           | 99.54       | 99.54       | 95.25              |
| CIFAR-10        | 94.65       | 99.32       | 88.16              |
| GTSRB           | 99.27       | 99.84       | 96.80              |


The results will be printed on screen.


## Contacts

If you have any questions, drop an email to _v.anhtt152@vinai.io_ , _v.anhnt479@vinai.io_  or leave a message below with GitHub (log-in is needed).
