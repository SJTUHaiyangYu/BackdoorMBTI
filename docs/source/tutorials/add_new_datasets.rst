Adding New Datasets
===================

To add a new dataset to BackdoorMBTI, follow the steps outlined below. The example provided uses CIFAR-100, but the same process can be applied to other datasets, including custom ones.

Dataset Loading Example
------------------------

In Backdoor, we use the function `load_dataset` (in `utils/data`) as our unified dataset loading interface. Here’s how to load a dataset (e.g., CIFAR-100):

.. code-block:: python

    def load_dataset(args, train=True):
        match args.dataset:
            case "cifar100":
                from torchvision.datasets import CIFAR100

                ds_dir = DATA_DIR / "CIFAR100"
                dataset = CIFAR100(
                    root=ds_dir,
                    train=train,
                    download=True,
                    transform=transforms.ToTensor(),
                )

Import the Dataset Module
--------------------------

If you are using a customed dataset, put it to the `datasets` folder, or the dataset is available in `torchvision.datasets`, import it. For example:

.. code-block:: python

    from torchvision.datasets import CIFAR100

Set the Dataset Path
---------------------

Specify the storage path for the dataset. By default, BackdoorMBTI uses the `./data` directory:

.. code-block:: python

    ds_dir = DATA_DIR / "CIFAR100"

Replace `"CIFAR100"` with your new dataset name.

Create the Dataset Instance
----------------------------

Use the imported dataset class to create the dataset instance, specifying whether it’s for training or testing, and ensure the data is downloaded if necessary:

.. code-block:: python

    dataset = CIFAR100(
        root=ds_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

Make sure the dataset outputs data in tensor format, as this is required for attack processing in BackdoorMBTI.

Call the Dataset
-----------------

You can now call the dataset instance in your code to load the training or test data.


.. code-block:: python

    from utils.data import load_dataset
    # or 
    from backdoormbti.utils.data import load_dataset

    dataset = load_dataset(args, train=True)

Following this structure, you can easily add new datasets to the project.
