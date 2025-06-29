Poison Dataset Wrapper
======================


The ``Poison Dataset Wrapper`` is implemented in the ``BackdoorMBTI/utils/data.py`` file and provides a class, ``BadSet``, for managing datasets with poisoned samples. This wrapper allows users to inject specific labels into a dataset at a controlled rate, making it especially useful for creating backdoor attacks in machine learning datasets.

.. code-block:: python

   class BadSet(Dataset):
       def __init__(
           self,
           benign_set,
           poison_set_path,
           type,
           dataset,
           attack,
           num_classes=None,
           mislabel=None,
           target_label=0,
           poison_rate=0.1,
           seed=0,
           mode=Literal["train", "test"],
           pop=False,
       ) -> None:
           ...

Class Parameters
----------------

- **benign_set**: The original, clean dataset.
- **poison_set_path**: Path to the poisoned dataset.
- **type**: Data type (e.g., "image", "text", "audio", "video").
- **dataset**: The name of the dataset being used.
- **attack**: Specifies the type of attack for which the poison data is created.
- **num_classes**: (Optional) Number of classes in the dataset.
- **mislabel**: (Optional) If `True`, randomly mislabels some benign samples.
- **target_label**: Default `0`. Specifies the target label for poisoned data. This is customizable to any label you want the poisoned samples to take.
- **poison_rate**: Default `0.1`. Specifies the proportion of data to poison within the dataset. For example, `poison_rate=0.2` means 20% of samples will be poisoned.
- **seed**: Default `0`. Random seed for reproducibility.
- **mode**: Indicates the dataset mode, either `"train"` or `"test"`.
- **pop**: If `True`, removes classes that do not match the target label from the poisoned set.


Main Methods
------------

**_get_poison_dataset**

Loads the poisoned dataset from the specified `poison_set_path`. Raises `FileNotFoundError` if the file is missing.

.. code-block:: python

   def _get_poison_dataset(self):
       """Load the poisoned dataset from a given path."""
       data_path = self.poison_set_path / f"{self.type}_{self.attack}_poison_{self.mode}_set.pt"
       if not data_path.exists():
           raise FileNotFoundError(f"No such file: {data_path}")
       return torch.load(data_path)


**_pop**

Removes classes from the dataset other than the target label. This is useful for focusing the dataset on the specific label being targeted in the poisoning attack.

.. code-block:: python

   def _pop(self):
       """Remove classes other than the target label from the dataset."""


**_mis_label**

For non-poisoned samples, randomly assigns an incorrect label. This method is particularly useful during training to add label noise.

.. code-block:: python

   def _mis_label(self, target, num_classes):
       """Randomly mislabel non-poisoned samples."""
       return (target + random.randint(1, num_classes)) % num_classes


**get_poisoned_index**

Determines which data samples to poison based on the `poison_rate` and `seed`. It returns a dictionary with the indices of poisoned samples.

.. code-block:: python

   def get_poisoned_index(self, length, seed, rate):
       """Calculate the indices for poisoned samples."""
       n = round(length * rate)
       torch.manual_seed(seed)
       indices = torch.randperm(length)[:n]
       return {int(idx): 1 for idx in indices}


**__getitem__**

Retrieves a data sample by index. If the sample is in the poisoned index or if the mode is "test", it assigns the `target_label` as the sample label. Supports different data types including image, text, audio, and video.

.. code-block:: python

   def __getitem__(self, index):
       """Retrieve a sample, applying target label if poisoned."""
       if index in self.poison_index or self.mode == "test":
           # Apply target label if poisoned
           return ...  # Logic for poisoned data
       else:
           return ...  # Logic for benign data


**__len__**

Returns the total length of the poisoned dataset.

.. code-block:: python

   def __len__(self):
       """Return the length of the poisoned dataset."""
       return len(self.poison_set)


Example Usage
-------------

Initialize the `BadSet` dataset with custom `target_label` and `poison_rate` values.

.. code-block:: python

   from BackdoorMBTI.utils.data import BadSet

   badset = BadSet(
       benign_set=original_dataset,
       poison_set_path="path/to/poison_set",
       type="image",
       dataset="example_dataset",
       attack="attack_type",
       target_label=1,     # Custom target label
       poison_rate=0.2     # Custom poison rate (20% of data)
   )

In this example, the `target_label` is set to `1`, and the `poison_rate` is `0.2`, meaning 20% of the data will be labeled as `1` to simulate a poisoning attack.
