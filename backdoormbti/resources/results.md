# Experimental Results

More results will be updated here soon!


- [Experimental Results](#experimental-results)
  - [Latest Progress](#latest-progress)
  - [Default Parameters](#default-parameters)
  - [The impact of low accuracy on backdoor attacks](#the-impact-of-low-accuracy-on-backdoor-attacks)
  - [The impact of low accuracy on backdoor defenses](#the-impact-of-low-accuracy-on-backdoor-defenses)
  - [The impact of noise level on backdoor attacks](#the-impact-of-noise-level-on-backdoor-attacks)
  - [The impact of noise level on backdoor defenses](#the-impact-of-noise-level-on-backdoor-defenses)
  - [The impact of poison ratio on backdoor attacks](#the-impact-of-poison-ratio-on-backdoor-attacks)
  - [The impact of poison ratio on backdoor defenses](#the-impact-of-poison-ratio-on-backdoor-defenses)
  - [The results of newly updated models](#the-results-of-newly-updated-models)
  - [The results of newly update modalities](#the-results-of-newly-update-modalities)
  - [The results of newly update defenses](#the-results-of-newly-update-defenses)



## Latest Progress

Here’s how you can organize the given items into a progress table for tracking:


| Experiments | Parameters | Total Experiments | Status | Description |
| --- | --- | --- | --- | --- |
| The impact of low accuracy on backdoor attacks | epochs=[1,2,5,10,20,30,40,50] | 9 | ✅ Completed |  |
| The impact of low accuracy on backdoor defenses | defenes=[STRIP, AC, FT, FP, ABL, CLP, NC] | 9(different accuracy)*7(defenses) | 🟢 In Progress |  |
| The impact of noise level on backdoor attacks | noise_level{μ=[-0.5,0,0.5], σ=[0.1,0.5,1]} | 9 | ✅ Completed |  |
| The impact of noise level on backdoor defenses | defenes=[STRIP, AC, FT, FP, ABL, CLP, NC] | 9(different noise level)*7(defenses) | 🟢 In Progress |  |
| The impact of poison ratio on backdoor attacks | pratio=[0.005, 0.01, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5] | 9 | ✅ Completed |  |
| The impact of poison ratio on backdoor defenses | defenes=[STRIP, AC, FT, FP, ABL, CLP, NC] | 9(different pratio)*7(7defenses) | 🟢 In Progress | We are collecting results from |
| The results of newly updated models | model=[ViT(dataset=CIFAR10,defense=fine-tuning), RoBERTa(dataset=SST-2), GPT2(dataset=SST-2), X-Vector(dataset=SpeechCommands), HuBert(dataset=speechcommands),  R3D(dataset=HMDB51)] | 16 | ✅ Completed | We select the 6 representative model to run first. |
| The results of newly updated modalities | modalities=[VQA(dataset=vqav2),Contrastive Learning(dataset=CIFAR10), Video(dataset=HMDB51)] | 3 | 🟢 In Progress |  |
| The results of newly updated defenses | attack=[BadNets],defense=[MNTD, FreeEagle] | 2 | 🟢 In Progress |  |

## Default Parameters

The default parameters are listed below.

| Parameter Name | Default Value | Description |
| --- | --- | --- |
| `dataset` | CIFAR10 | The default dataset used for training and evaluation (e.g., CIFAR10) |
| `attack` | BadNets | Type of backdoor attack being used as default (e.g., BadNets) |
| `model` | resnet18 | The default model architecture used for experiments (e.g., ResNet18) |
| `pratio` | 0.1 | Poison ratio, representing the proportion of poisoned data |
| `noise level` | none | The default level of noise added to the data (e.g., none for clean data) |
| `noise percentage` | 25% | The default percentage of noise added, when applicable (e.g., 25%) |
| `defense` | STRIP | Defense method used as default to mitigate backdoor attacks (e.g., STRIP) |


This table provides a clear breakdown of ongoing, planned, and completed tasks for easy tracking and updates.




## The impact of low accuracy on backdoor attacks
<a href="#top">[Back to top]</a>

We obtained different ACCs by adjusting various epochs, and below are the results of our attack experiments. The results show that different ACCs do not affect ASR, which aligns with the theory of backdoor learning quick learning.
|epochs|clean accuracy |attack success rate |
| -------- | -------- | -------- |
|1| 0.4941 |0.9996 |
|2| 0.5421| 0.9998 | 
|5| 0.6938| 0.9997|
|10|0.7501| 1.0000|
|20| 0.7900| 1.0000|
|30|0.8164|0.9996|
|40|0.8264|0.9997|
|50|0.8377|1.0000|

## The impact of low accuracy on backdoor defenses
<a href="#top">[Back to top]</a>

We test low accuracy on fine-tuning, and the results are below.The lower initial accuracy similarly has no significant impact on fine-tuning
|fine-tuning_epochs|atk_train_epochs|clean accuracy |attack success rate |
|-------| -------- | -------- | -------- |
|10|1| 0.7713 |0.0031 |
|10|2| 0.7820| 0.0194 | 
|10|5|0.8019| 0.0057|
|10|10|0.8183| 0.0103|
|10|20| 0.8343| 0.0078|
|10|30|0.8398|0.0385|
|10|40|0.8468|0.02766|
|10|50|0.8409|0.0175|
## The impact of noise level on backdoor attacks
<a href="#top">[Back to top]</a>

We obtained different levels of noise by adjusting the mean and variance, and below are the results of our attack experiments. Through the experiments, we found that moderate levels of noise have almost no impact on backdoor attacks.
| mean| variance | attack success rate |attack success rate |
| -------- | -------- | -------- | -------- |
|-0.5|0.1|0.8328|0.9998|
|-0.5|0.5|0.8248|0.9996|
|-0.5|1.0|0.8241|0.9997|
|0.0|0.1|0.8356|0.9990|
|0.0|0.5|0.8274|0.9997|
|0.0|1.0|0.8185|1.0000|
|0.5|0.1|0.8274|0.9997|
|0.5|0.5|0.8287|0.9996|
|0.5|1.0|0.8212|0.9998|
## The impact of noise level on backdoor defenses
<a href="#top">[Back to top]</a>

We test different levels of noise on fine-tuning, and the results are below.Different levels of noise similarly have no significant impact on fine-tuning
|fine-tuning_epochs| mean| variance | attack success rate |attack success rate |
| -------- | -------- | -------- | -------- | -------- |
|10|-0.5|0.1|0.7990|0.1174|
|10|-0.5|0.5|0.8008| 0.1104|
|10|-0.5|1.0|0.7972|0.1282|
|10|0.0|0.1|0.7979|0.0727|
|10|0.0|0.5|0.8019|0.2118|
|10|0.0|1.0|0.7998|0.0517|
|10|0.5|0.1|0.8052|0.0615|
|10|0.5|0.5|0.8018|0.2092|
|10|0.5|1.0|0.7962|0.1216|

## The impact of poison ratio on backdoor attacks
<a href="#top">[Back to top]</a>

The experimental results are shown in the table below. When the poisoning rate is low, increasing the poisoning rate can significantly improve the attack accuracy. However, once the poisoning rate reaches a certain level (0.08), further increasing the poisoning rate has little effect.
| pratio | clean accuracy | attack success rate |
| -------- | -------- | -------- |
| 0.005 | 0.7746 | 0.754 |
|0.01 | 0.7756| 0.8529 |
| 0.04 | 0.768|0.9268 |
| 0.08 | 0.7669|0.9486 |
| 0.1 | 0.7633|0.9574|
| 0.2 | 0.7572|0.9684 |
|0.3 | 0.7351|0.9762|
|0.4 | 0.7208|0.9811|
|0.5 | 0.6984|0.985|


## The impact of poison ratio on backdoor defenses
<a href="#top">[Back to top]</a>

we test the impact of poison ratio on fine-tuning, and the results are below.
|fine-tuning_epochs| pratio | clean accuracy | attack success rate |
| -------- | -------- | -------- | -------- |
|10| 0.005 |0.8184 | 0.0045 |
|10|0.01 | 0.8163| 0.0037|
|10| 0.04 | 0.8109|0.0057 |
|10| 0.08 | 0.8177|0.0037 |
|10| 0.1 | 0.8350|0.0192|
|10| 0.2 | 0.8120|0.0055 |
|10|0.3 | 0.8065|0.0118|
|10|0.4 | 0.8113|0.0035|
|10|0.5 | 0.8041|0.0063|
## The results of newly updated models
<a href="#top">[Back to top]</a>

For image, we have conducted evaluations on the newly updated models using the CIFAR-10 dataset with a total of 20 epochs, employing the BadNets attack (poison rate 10%) as the default attack. Below are the completed results:

| Model | Accuracy | Attack Success Rate |
| --- | --- | --- |
| ViT | 0.9678 | 1 |
| resnet18 | 0.71 | 0.96 |
| resnet34 | 0.7998 | 0.6934 |
| resnet50 | 0.6118 | 0.7125 |
| densenet121 | 0.8385 | 0.7804 |
| densenet161 | 0.8104 | 0.7238 |
| mobilenet_v2 | 0.8544 | 0.9997 |
| vgg11 | 0.8891 | 0.1605 |
| vgg16 | 0.79 | 0.95 |
| vgg19 | 0.8654 | 0.9551 |
| shufflenet_v2 | 0.7034 | 0.7686 |
| efficientnet | 0.3862 | 0.1455 |

For audio, we conducted evaluations on HuBERT using the SpeechCommands dataset, training for 20 epochs with a BadNets attack at a 10% poison rate. The results are as follows:

| Model | Accuracy | Attack Success Rate |
| --- | --- | --- |
| HuBERT | 0.9758 | 0.9943 |
|LSTM	|0.754	|0.9921|
|VGGVox|	0.9416|	0.9978|

For text, we trained GPT-2 and RoBERTa on the SST-2 dataset for 20 epochs, also using a BadNets attack with a 10% poison rate. The results are as follows:

| Model | Accuracy | Attack Success Rate |
| --- | --- | --- |
| GPT-2 | 0.8977 | 1 |
| RoBERTa | 0.8988 | 1 |

## The results of newly update modalities
<a href="#top">[Back to top]</a>

For contrastive learning, we trained SimCLR using ResNet-18 as the backbone on the CIFAR-10 dataset and tested it on the STL-10 dataset over 200 epochs. A BadEncoder attack with a poison rate of 5e-4 was applied. The results are as follows:

| Model | Accuracy | Attack Success Rate |
| --- | --- | --- |
| SimCLR | 0.9758 | 0.9943 |

## The results of newly update defenses
We added two backdoor detection methods. The table below shows their detection results.

<a href="#top">[Back to top]</a>
|Model|Defense|Category|AUC|
| --- | --- | --- |---|
|ResNet18|MNTD|Detect Backdoored Models|0.9994|
