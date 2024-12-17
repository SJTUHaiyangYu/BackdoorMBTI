Adding New Defenses
===================

This guide will walk you through the process of adding a defense method similar to ``STRIP``, which detects poisoned data by analyzing the entropy of perturbed samples. To create a similar defense class, you will define a new defense method or enhance an existing one while following the same structure.

Here we show the `STRIP` defense as an example to illustrate how to add a new defense method. The steps are as follows:

Define the Defense Class
------------------------

To create a new defense, you need to inherit from `InputFilteringBase`, which provides basic filtering functionality. In the constructor (``__init__``), set the relevant attributes, such as the type of data the defense handles (e.g., images, text, audio) and other defense-specific parameters.

Example:

.. code-block:: python

    class STRIP(InputFilteringBase):
        def __init__(self, args) -> None:
            super().__init__(args)
            self.args = args
            self.set_pertub_func()  # Set the perturbation function based on data type

Define the Perturbation Functions
---------------------------------

The `STRIP` defense works by perturbing inputs (e.g., adding noise to images, shuffling words in text) and then measuring the entropy of the model's output on these perturbed samples. The key is to define how to perturb different types of data.

Example code for image perturbation:

.. code-block:: python

    def perturb_img(self, img):
        perturbation = torch.randn(img.shape)
        return img + perturbation  # Add noise to the image

Example code for text perturbation:

.. code-block:: python

    def perturb_txt(self, text):
        words = text.split()
        swap_pos = np.random.randint(0, len(words), int(len(words) * 0.1))
        for idx in swap_pos:
            words[idx] = self.replace_words[np.random.randint(len(self.replace_words))]
        return " ".join(words)  # Randomly replace some words

Calculate Entropy for Defense
-----------------------------

The core of the defense is calculating the entropy of the model's predictions on perturbed data. Based on this entropy, the defense decides whether the input is poisoned or clean.

Example:

.. code-block:: python

    def cal_entropy(self, model, data_lst, sample=False):
        perturbed_samples = [self.perturb(data) for data in data_lst]
        probs = model(perturbed_samples)
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1)  # Compute entropy
        return entropy

Implement the Sample Filtering Method
-------------------------------------

- Write the `sample_filter()` method to compute entropy and determine whether the sample is malicious based on a pre-defined threshold.
- If the entropy is below the threshold, mark the sample as malicious.

Example:

.. code-block:: python

    def sample_filter(self, data):
        poison_entropy = self.cal_entropy(self.model.model, data, sample=True)
        return (1 if poison_entropy < self.threshold else 0, poison_entropy)  # 1 for malicious

Test the Defense
----------------

After defining your defense method, it is important to test it under various scenarios and datasets. Make sure to verify that the filtering logic correctly identifies malicious samples across different data types (e.g., images, text, audio).

Run your defense with different thresholds and configurations to optimize performance.
