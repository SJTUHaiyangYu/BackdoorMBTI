Defenses
========

Supported Defenses
------------------

The following table lists the supported defenses in BackdoorMBTI:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Defense
     - Modality
     - Input
     - Stage
     - Output
     - Paper
   * - STRIP
     - Audio, Image, and text
     - backdoor model, clean dataset
     - post-training
     - clean dataset
     - `STRIP: A Defence Against Trojan Attacks on Deep Neural Networks <https://arxiv.org/pdf/1902.06531.pdf>`__
   * - AC
     - Audio, Image, and text
     - backdoor model, clean dataset, poison dataset
     - post-training
     - clean model, clean dataset
     - `Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering <https://arxiv.org/pdf/1811.03728.pdf>`__
   * - FT
     - Audio, Image, and text
     - backdoor model, clean dataset
     - in-training
     - clean model
     - `Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks <https://arxiv.org/pdf/1805.12185.pdf>`__
   * - FP
     - Audio, Image, and text
     - backdoor model, clean dataset
     - post-training
     - clean model
     - `Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks <https://arxiv.org/pdf/1805.12185.pdf>`__
   * - ABL
     - Audio, Image, and text
     - backdoor model, poison dataset
     - in-training
     - clean model
     - `Anti-Backdoor Learning: Training Clean Models on Poisoned Data <https://arxiv.org/pdf/2110.11571.pdf>`__
   * - CLP
     - Audio, Image, and text
     - backdoor model
     - post-training
     - clean model
     - `Data-free Backdoor Removal based on Channel Lipschitzness <https://arxiv.org/pdf/2208.03111.pdf>`__
   * - NC
     - Image
     - backdoor model, clean dataset
     - post-training
     - clean model, trigger pattern
     - `Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks <https://par.nsf.gov/servlets/purl/10120302>`__
   * - MNTD
     - Image
     - backdoor model
     - post-training
     - detection result
     - `Detecting AI Trojans  Using Meta Neural Analysis <https://arxiv.org/pdf/1910.03137>`__
   * - FreeEagle
     - Image
     - backdoor model
     - post-training
     - detection result
     - `FREEEAGLE: Detecting Complex Neural Trojans in Data-Free Cases <https://www.usenix.org/system/files/usenixsecurity23-fu-chong.pdf>`__
