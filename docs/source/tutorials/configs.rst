Configs
=======

The `Configs` directory contains various configuration files and settings used for attacks, defenses, and training in the backdoor learning system. This directory is organized into subdirectories and files, each serving a specific purpose.

Directory Structure
-------------------

.. code-block:: text

    configs/
    ├── attacks/
    │   ├── audio/
    │   ├── image/
    │   ├── text/
    │   └── video/
    ├── defenses/
    ├── __init__.py
    ├── default.yaml
    ├── settings.py
    ├── train_audio_args.yaml
    ├── train_image_args.yaml
    ├── train_text_args.yaml
    └── train_video_args.yaml

attacks
-------

The `attacks` subdirectory contains configuration files specific to different types of data attacks, organized by data type: `audio`, `image`, `text`, and `video`. Each folder contains YAML configuration files defining parameters and settings for various backdoor attack methods in the respective data domains.

defenses
--------

The `defenses` subdirectory includes configuration files for defense mechanisms against backdoor attacks. These files outline settings for different defense strategies to mitigate the effects of data poisoning in various data types.

Configuration Files
-------------------

settings.py
~~~~~~~~~~~

The `settings.py` file defines key directories and lists of available attacks, targets, datasets, models, and defenses, as well as configurations like poison rates. This file serves as a central location for defining paths and lists for commonly used components in the system.

**Directory Paths**:

- ``BASE_DIR``: The base directory of the project.
- ``DATA_DIR``: Directory where data is stored.
- ``LOG_DIR``: Directory for logging information.
- ``TEST_DIR``: Directory for testing files.
- ``POISON_DATA_DIR``: Directory for poisoned data storage.

**TYPES**: Specifies supported data types for backdoor attacks, including ``image``, ``text``, and ``audio``.

**ATTACKS**: Defines supported attack types for each data type:

- **image**: Includes attack methods like ``badnet``, ``blend``, ``bpp``, ``wanet``, and others.
- **text**: Includes attacks like ``badnet``, ``addsent``, and ``stylebkd``.
- **audio**: Includes attacks such as ``badnet``, ``blend``, ``gis``, and ``ultrasonic``.
- **video**: Limited to ``badnet`` and ``tuap`` for video data.

**TARGETS**: Specifies target labels for each dataset used in attack setups, e.g., "jazz" for ``gtzan`` or ``0`` for ``voxceleb1identification``.

**DATASETS**: Defines available datasets for each data type:

- **image**: e.g., ``cifar10``, ``gtsrb``, ``celeba``.
- **text**: e.g., ``imdb``, ``dbpedia``, ``sst2``.
- **audio**: e.g., ``speechcommands``, ``gtzan``, ``voxceleb1identification``.
- **video**: e.g., ``hmdb51``.

**MODELS**: Lists supported models for each data type, such as:

- **image**: ``resnet18``, ``vit_b_16``.
- **text**: ``bert``, ``gpt2``, ``roberta``.
- **audio**: ``audiocnn``, ``lstm``.
- **video**: ``r3d``.

**DEFENSES**: Contains a list of available defense mechanisms, including ``ac``, ``strip``, ``finetune``, ``nc``, ``onion``, and others.

**POISON_RATE_LST**: Lists various poisoning rates to test, ranging from ``0.000`` to ``0.5``.

default.yaml
~~~~~~~~~~~~

The `default.yaml` file defines the default settings for training and testing, including model configurations, dataset information, and other training parameters.

**Key Parameters**:

- **attack_label_trans**: Type of label transformation for the attack, e.g., ``all2one``.
- **attack_target**: Target label for the attack, default is ``0``.
- **client_optimizer**: Optimizer type, e.g., ``sgd``.
- **dataset**: Dataset to be used, e.g., ``cifar10``.
- **frequency_save**: Frequency to save model checkpoints.
- **batch_size**: Batch size for training, default ``128``.
- **lr**: Learning rate, default ``0.01``.
- **model**: Model architecture, e.g., ``resnet18``.
- **pratio**: Poisoning ratio, default ``0.1``.
- **epochs**: Number of training epochs, default ``100``.

train_audio_args.yaml, train_image_args.yaml, train_text_args.yaml, train_video_args.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These YAML files provide specific configurations for training with different data types: ``audio``, ``image``, ``text``, and ``video``. Each file includes parameters such as model type, number of epochs, batch size, learning rate, and optimizer settings. They allow customized settings based on the data type being used in training.

- **model**: Model architecture for each data type, e.g., ``audiocnn`` for ``audio``.
- **epochs**: Number of epochs for training.
- **batch_size**: Batch size for training.
- **client_optimizer**: Optimizer type, typically ``sgd``.
- **lr**: Learning rate.
- **weight_decay**: Weight decay for regularization.
