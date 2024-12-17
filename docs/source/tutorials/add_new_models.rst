Adding New Models
=================

This guide explains how to define and register custom models, as well as how to add data preprocessing transformations when necessary. If you're using pre-built models from `torchvision.models` or `transformers`, no additional model definition is required, just register the model.


Model Definition
----------------

To define a custom model architecture, add the model's architecture definition in the appropriate folder, such as `model`. If you are using pre-defined models from libraries like `torchvision.models` or `transformers`, you can skip this step and directly proceed to the model registration.

Steps:

1. Navigate to the `model` folder.
2. Create or update the necessary Python file with your model architecture.
3. Ensure that the model class inherits from the appropriate base class (e.g., `torch.nn.Module`).

For example, if you are defining a PyTorch model:

.. code-block:: python

    import torch
    import torch.nn as nn

    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.fc1 = nn.Linear(32*28*28, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x


Registering the Model
---------------------

To register the custom model for later use, you need to instantiate the model in the `load_model` function located in the `utils/model.py` file.

Steps:

1. Open `utils/model.py`.
2. Locate the `load_model` function.
3. Add code to instantiate your model.

For example, to register a model named `CustomModel`:

.. code-block:: python

    from model.custom_model import CustomModel  # Adjust the import path accordingly

    def load_model(args, **kwargs):
        match args.model:
            case "CustomModel":
                model = CustomModel(your_arguments)
                # if need data pre_transform, see below
                pre_transform = None
                args.pre_trans = pre_transform
            # torchvision 
            case "googlenet":
                from torchvision.models import googlenet

                model = googlenet(num_classes=args.num_classes)
            # transformers 
            case "gpt2":
                from transformers import (
                  AutoConfig,
                  AutoModelForSequenceClassification,
                  AutoTokenizer,
                )
                # load pretrained model
                model_config = AutoConfig.from_pretrained(model_path)
                model_config.num_labels = args.num_classes
                model_config.update(kwargs)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    config=model_config,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, model_max_length=512
                )
                args.tokenizer = tokenizer
            # Add more models here as needed
            case _:
                  raise NotImplementedError("Model %s not supported." % args.model)
        return model

(Optinal) Add Data PreTransforms
--------------------------------

If your model requires the data to be in a specific format or shape, you may need to add a preprocessing step, known as a "pre_transform". This can be done to ensure that the input data is compatible with the model's requirements.

To add a pre-transform, follow these steps:

1. Identify the preprocessing operations needed for your model (e.g., resizing images, normalizing data, or converting data types).
2. Implement these operations in a preprocessing function or pipeline.
3. Call this preprocessing step before feeding the data into the model.

For example, if your model requires input images to be resized to 224x224 and normalized:

.. code-block:: python

    from torchvision import transforms

    def pre_transform():
        # define your preprocessing steps here
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


By using pre-transforms, ensure that the input data fits the model's expected format.
