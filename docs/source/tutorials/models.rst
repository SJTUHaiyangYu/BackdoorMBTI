Models
======

This section provides a brief overview of the models used in this project, categorized by task type.

Text Models
------------

.. list-table:: Available Text Models
   :header-rows: 1

   * - Model
     - Description
   * - BERT
     - A transformer model that excels at tasks like text classification, QA, and more.
   * - RoBERTa
     - An optimized variant of BERT with improved training techniques for better NLP performance.
   * - GPT-2
     - A large-scale model known for text generation, capable of producing coherent text.

Image Models
------------

.. list-table:: Available Image Models
   :header-rows: 1

   * - Model
     - Description
   * - ResNet18/34/50
     - A family of CNNs with 18, 34, and 50 layers respectively, utilizing residual connections to improve training in deep networks.
   * - DenseNet121/161
     - CNNs where each layer is connected to every other layer, reducing parameters while maintaining high accuracy in image classification tasks.
   * - MobileNetV2
     - A lightweight CNN optimized for mobile and resource-constrained environments, effective for image classification.
   * - InceptionV3
     - Known for its inception modules, this model efficiently handles multi-scale features for image classification.
   * - GoogleNet
     - Similar to InceptionV3, GoogleNet uses inception modules and is designed for efficient computation.
   * - ShuffleNetV2_x1_0
     - A lightweight CNN designed for fast computation on mobile devices, balancing accuracy and efficiency.
   * - EfficientNet-B0
     - Part of the EfficientNet family, this model scales depth, width, and resolution to achieve high performance on image tasks.
   * - AlexNet
     - One of the earliest deep CNNs that popularized deep learning, effective in image classification tasks.
   * - VGG11/16/19
     - A set of deep CNNs with 11, 16, or 19 layers, known for their simplicity and effectiveness in image classification.
   * - Vision Transformer (ViT-B_16)
     - A transformer model applied to image classification, treating images as sequences of patches instead of traditional convolutions.
   * - R3D (ResNet3D)
     - A 3D CNN for video classification tasks, extending 2D convolutions to three dimensions to handle spatial and temporal information.

Audio Models
------------

.. list-table:: Available Audio Models
   :header-rows: 1

   * - Model
     - Description
   * - Hubert
     - A transformer model designed for speech recognition and audio classification tasks, using self-supervised learning on audio data.
   * - AudioCNN
     - A CNN specifically designed to process and classify raw audio signals.
   * - AudioLSTM
     - An LSTM-based model tailored for sequential audio data, used in tasks like speech recognition and audio classification.
   * - X-Vector
     - A model used for speaker verification, embedding speaker identity for classification.
   * - VGGVox
     - A CNN-based model for speaker recognition tasks, adapted from the VGG architecture for audio inputs.
   * - SpeechEmbedder
     - A model that extracts speaker embeddings from audio, used in speaker verification tasks.
