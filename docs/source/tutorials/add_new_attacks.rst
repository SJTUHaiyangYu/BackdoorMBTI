Adding New Attacks
===================

This guide will walk you through the process of adding an attack similar to ``BadNet``, an image-based backdoor attack, which poisons image data by overlaying a patch trigger. To create a similar class, you will define a new backdoor method or enhance an existing one while following the same structure.

Here we show the badnet attack as an example to illustrate how to add a new attack. The steps are as follows:

.. code-block:: python

    class BadNet(ImageBase):
        def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
            super().__init__(dataset, args, mode, pop)
            self.attack_type = "image"
            self.attack_name = "badnet"

            # define poison image transformer
            trans = transforms.Compose(
                [
                    transforms.Resize(self.args.input_size[:2], antialias=True),
                    transforms.ToTensor(),
                ]
            )
            trigger_path = Path(self.args.patch_mask_path)
            self.bd_transform = AddMaskPatchTrigger(
                trans(Image.open(BASE_DIR / trigger_path))
            )

        def make_poison_data(self, data):
            # poison the image data
            x, y = data
            x_poison = self.bd_transform(x)
            # set mislabel
            y_poison = self.args.attack_target
            is_poison = 1
            y_original = y
            return (x_poison, y_poison, is_poison, y_original)

Define the Attack Class
--------------------------------------------------

   - Inherit from the base class ``ImageBase``, and it's a ``LightningModule``.
   - Add relevant parameters:
        - dataset: the clean dataset
        - mode: indicate you are training or testing 
        - pop: whether to pop the original ``attack_targe`` class from the dataset
   - Inside the ``__init__`` method, set up key attributes like ``attack_type`` and ``attack_name``, which describe the attack.

   Example code:

   .. code-block:: python

      class BadNet(ImageBase):
          def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
              super().__init__(dataset, args, mode, pop)
              self.attack_type = "image"  # Set the attack type
              self.attack_name = "badnet"  # Name your custom attack

Define the Poison Data Transform Function
-----------------------------------------

The poison data transformation is a critical step in generating poisoned data. The transform function involves applying a trigger to the input image to create a poisoned data for BadNets.

However for more sophisticated attack method, the transform function can be more complex. For example, Label-Consistent backdoor attacks using GAN to generate the poisoned data. In this case, the transform function can be GAN model to generate the poisoned data.

Here we using BadNet as example to illustrate how to define the poison data transform function:

   - Load the custom trigger (e.g., a patch or any other backdoor trigger) using ``Image.open()`` and apply it to the transformation.
   - The ``AddMaskPatchTrigger`` is a placeholder for the specific backdoor technique you are implementing. You may need to define or modify this function for your needs.

   Example code:

   .. code-block:: python

          # Define poison image transformer
          trigger_path = Path(self.args.patch_mask_path)
          self.bd_transform = AddMaskPatchTrigger(
              trans(Image.open(BASE_DIR / trigger_path))
          )

Implement the Poison Data Method
--------------------------------

   - Write the ``make_poison_data()`` method to modify the input data and generate poisoned versions. This function will be called by ``make_and_save_poisoned_data()`` in the base class.
   - Use the backdoor transformer (``self.bd_transform``) to apply the trigger to the input image ``x``.
   - Assign the poisoned label to ``y_poison`` and track whether the data has been poisoned with a flag ``is_poison``.
   - The method should return a tuple following the format:  (poisoned_image, target_label, poisoning_flag, original_label).

   Example code:

   .. code-block:: python

          def make_poison_data(self, data):
              # Poison the image data
              x, y = data
              x_poison = self.bd_transform(x)  # Apply the backdoor trigger
              y_poison = self.args.attack_target  # Assign the target label
              is_poison = 1  # Mark the data as poisoned
              y_original = y  # Keep the original label
              return (x_poison, y_poison, is_poison, y_original)

Testing
-------------------------

   - Customize the trigger application or transformation based on the type of attack you are developing.
   - Test your new attack by passing different datasets and configurations to ensure the poison data is created correctly.
