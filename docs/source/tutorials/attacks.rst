Attacks
=======

Supported Attacks
-----------------

The following table lists the supported attacks in BackdoorMBTI:

.. list-table:: 
   :header-rows: 1

   * - Modality
     - Attack
     - Visible
     - Pattern
     - Add
     - Sample Specific
     - paper
   * - Image
     - AdaptiveBlend
     - Invisible
     - Global
     - Yes
     - No
     - `REVISITING THE ASSUMPTION OF LATENT SEPARABILITY FOR BACKDOOR DEFENSES <https://openreview.net/pdf?id=_wSHsgrVali>`_
   * - Image
     - BadNets
     - Visible
     - Local
     - Yes
     - No
     - `Badnets: Evaluating backdooring attacks on deep neural networks <https://ieeexplore.ieee.org/iel7/6287639/8600701/08685687.pdf>`_
   * - Image
     - Blend
     - Invisible
     - Global
     - Yes
     - Yes
     - `Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning <https://arxiv.org/abs/1712.05526v1>`_
   * - Image
     - Blind(under test)
     - Visible
     - Local
     - Yes
     - Yes
     - `Blind Backdoors in Deep Learning Models <https://www.cs.cornell.edu/~shmat/shmat_usenix21blind.pdf>`_
   * - Image
     - BPP
     - Invisible
     - Global
     - Yes
     - No
     - `Bppattack: Stealthy and efficient trojan attacks against deep neural networks via image quantization and contrastive adversarial learning <http://openaccess.thecvf.com/content/CVPR2022/papers/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.pdf>`_
   * - Image
     - DynaTrigger
     - Visible
     - Local
     - Yes
     - Yes
     - `Dynamic backdoor attacks against machine learning models <https://arxiv.org/pdf/2003.03675>`_
   * - Image
     - EMBTROJAN(under test)
     - Invisible
     - Local
     - Yes
     - No
     - `An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks <https://dl.acm.org/doi/pdf/10.1145/3394486.3403064>`_
   * - Image
     - LC
     - Invisible
     - Global
     - No
     - Yes
     - `Label-consistent backdoor attacks <https://openaccess.thecvf.com/content/ICCV2021/papers/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.pdf>`_
   * - Image
     - Lowfreq
     - Invisible
     - Global
     - Yes
     - Yes
     - `Rethinking the Backdoor Attacks’ Triggers: A Frequency Perspective <https://arxiv.org/pdf/1912.02771/>`_
   * - Image
     - PNoise
     - Invisible
     - Global
     - Yes
     - Yes
     - `Use procedural noise to achieve backdoor attack <https://ieeexplore.ieee.org/iel7/6287639/9312710/09529206.pdf>`_
   * - Image
     - Refool
     - Invisible
     - Global
     - Yes
     - No
     - `Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550188.pdf>`_
   * - Image
     - SBAT
     - Invisible
     - Global
     - No
     - Yes
     - `Stealthy Backdoor Attack with Adversarial Training <https://ieeexplore.ieee.org/abstract/document/9746008/>`_
   * - Image
     - SIG
     - Invisible
     - Global
     - Yes
     - No
     - `A NEW BACKDOOR ATTACK IN CNNS BY TRAINING SET CORRUPTION WITHOUT LABEL POISONING <https://arxiv.org/pdf/1902.11237>`_
   * - Image
     - SSBA
     - Invisible
     - Global
     - No
     - Yes
     - `Invisible Backdoor Attack with Sample-Specific Triggers <https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf>`_
   * - Image
     - trojanNN(under test)
     - Visible
     - Local
     - Yes
     - Yes
     - `Trojaning Attack on Neural Network <https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech>`_
   * - Image
     - ubw
     - Invisible
     - Global
     - Yes
     - No
     - `Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection <https://proceedings.neurips.cc/paper_files/paper/2022/file/55bfedfd31489e5ae83c9ce8eec7b0e1-Paper-Conference.pdf>`_
   * - Image
     - WaNet
     - Invisible
     - Global
     - No
     - Yes
     - `WaNet -- Imperceptible Warping-Based Backdoor Attack <https://arxiv.org/pdf/2102.10369>`_
   * - Text
     - AddSent
     - Visible
     - Local
     - Yes
     - No
     - `A backdoor attack against LSTM-based text classification systems <https://arxiv.org/pdf/1905.12457.pdf>`_
   * - Text
     - BadNets
     - Visible
     - Local
     - Yes
     - No
     - `Badnets: Evaluating backdooring attacks on deep neural networks <https://ieeexplore.ieee.org/iel7/6287639/8600701/08685687.pdf>`_
   * - Text
     - BITE
     - Invisible
     - Local
     - Yes
     - Yes
     - `Textual backdoor attacks with iterative trigger injection <https://u1x3881ofs0.feishu.cn/sheets/VHbrsq8MdhV7BPtd77Nc6BGSnIc?sheet=ae56f0&range=QTE4>`_
   * - Text
     - LWP
     - Visible
     - Local
     - Yes
     - No
     - `Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning <https://aclanthology.org/2021.emnlp-main.241.pdf>`_
   * - Text
     - STYLEBKD
     - Visible
     - Global
     - No
     - Yes
     - `Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer <https://arxiv.org/pdf/2110.07139>`_
   * - Text
     - SYNBKD
     - Invisible
     - Global
     - No
     - Yes
     - `Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger <https://arxiv.org/pdf/2105.12400.pdf>`_
   * - Audio
     - Baasv(under test)
     - \-
     - Global
     - Yes
     - No
     - `Backdoor Attack against Speaker Verification <https://arxiv.org/pdf/2010.11607>`_
   * - Audio
     - Blend
     - \-
     - Local
     - Yes
     - No
     - `Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning <https://arxiv.org/abs/1712.05526v1>`_
   * - Audio
     - DABA
     - \-
     - Global
     - Yes
     - No
     - `Opportunistic Backdoor Attacks: Exploring Human-imperceptible Vulnerabilities on Speech Recognition Systems <https://dl.acm.org/doi/abs/10.1145/3503161.3548261>`_
   * - Audio
     - GIS
     - \-
     - Global
     - No
     - No
     - `Going in style: Audio backdoors through stylistic transformations <https://arxiv.org/pdf/2211.03117>`_
   * - Audio
     - UltraSonic
     - \-
     - Local
     - Yes
     - No
     - `Can You Hear It? Backdoor Attacks via Ultrasonic Triggers <https://github.com/skoffas/ultrasonic_backdoor>`_
