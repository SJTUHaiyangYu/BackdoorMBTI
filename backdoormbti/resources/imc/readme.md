# imc

A tale of evil twins: Adversarial inputs versus poisoned models 

由于源工作已经嵌入到TrojanZoo框架中，已经相对完整，有多种攻击实现，数据结构的定义模式多且依赖关系复杂，难以解耦。

再加上所阅读的论文本身提出的是输入与模型联合两阶段优化的攻击框架，模型框架自由度高，主要涉及loss修正、训练流程优化、攻击模型适配等，进一步增大了解耦难度。

为了保障源工作良好的性能展示，本嵌入工作主体通过自定义imc.yaml的形式将trojanzoo框架所需的参数动态传入进行执行，重点在于数据集以及模型框架的路径设置

## 流程
1. 安装trojanzoo库
pip install trojanzoo
2. 设置imc.yaml的参数
其中的可用参数均已展示，sh文件可以动态读取生成命令
3. 数据集以及模型路径适配
注意的是，作者框架的路径定义为粗目录，并不是具体的数据集或模型目录，比如：

数据集目录指定为data/data/，则image、cifar10的最终目录为resource/imc/data/data/image/cifar10/

数据集目录指定为data/model/，则image、resnet18_comp.pth的最终目录为resource/imc/data/model/image/cifar10/

对于特定的trojanNN等水印触发器攻击，可提供的水印目录在当前目录直接显示，使用时通过--mask_path指定水印名即可

5. 运行
python ./atk_train.py --data_type image --dataset cifar10 --attack_name IMC --model resnet18 --pratio 0.1 --num_workers 4
6. 结果
结果以及优化的攻击模型与攻击水印均在resource/imc/data/attack下
