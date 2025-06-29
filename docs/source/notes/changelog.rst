Change Logs
===========

2024-10-13
-----------
- **Initialized documentation framework**: Established the overall structure for the Read the Docs documentation.

  - Created main sections under three categories:

    - **Tutorials**: Comprehensive guides and tutorials for users.
    - **API Documents**: Placeholder for detailed API documentation.
    - **Notes**: Additional information and project notes.

  - Created initial ``.rst`` files for each section:

    - **Tutorials**:
    
      - ``install.rst``: Installation instructions.
      - ``getting_started.rst``: Quick start guide.
      - ``tasks.rst``: Overview of supported tasks.
      - ``datasets.rst``: Information on available datasets.
      - ``adding_new_datasets.rst``: Guide for adding new datasets.
      - ``attacks.rst``: Description of available attack methods.
      - ``adding_new_attacks.rst``: Guide for adding new attack methods.
      - ``poison_dataset_wrapper.rst``: Documentation for the poison dataset wrapper.
      - ``models.rst``: Overview of supported models.
      - ``adding_new_models.rst``: Guide for adding custom models.
      - ``defenses.rst``: Description of defense mechanisms.
      - ``adding_new_defenses.rst``: Guide for adding new defenses.
      - ``configs.rst``: Configuration file guidelines.

    - **API Documents**:
    
      - ``api_reference.rst``: Placeholder for API reference documentation.

    - **Notes**:
    
      - ``scalability.rst``: Information on scalability strategies.
      - ``contributing.rst``: Guidelines for contributing to BackdoorMBTI.
      - ``changelog.rst``: Changelog file to track documentation updates.

  - **Updated index.rst**: Linked all newly created ``.rst`` files in ``index.rst`` to provide a structured navigation within the documentation.


2024-10-15
-----------
- **Updated document framework**: Corrected minor errors in the overall documentation structure.
- **Added content to** ``contributing.rst``: Included a guide to help new contributors get started with BackdoorMBTI.
- **Added content to** ``models.rst``: Documented instructions on defining and registering custom models, along with data preprocessing transformations.
  

2024-10-16
-----------
- **Added content to** ``getting_started.rst``: Created a quick start guide for BackdoorMBTI, including steps for downloading data, running backdoor attacks, and executing defense experiments.
- **Added content to** ``tasks.rst``: Documented the various tasks supported by BackdoorMBTI, including new tasks like video, audiovisual, VQA, and contrastive learning.
- **Modified** ``models.rst``: Updated the ``def load_model(args):`` function to improve functionality.
- **Updated titles**: Adjusted and standardized section titles across the documentation.
- **Revised** ``getting_started.rst``: Made additional updates to the content for clarity and completeness.
- **Fixed error in** ``install.rst``: Corrected minor errors in the installation guide.


2024-10-17
-----------
- **Updated** ``getting_started.rst``: Continued making improvements to the quick start guide.
- **Modified** ``tasks.rst``: Added descriptions for new tasks: Contrastive Learning (CL) and Visual Question Answering (VQA).
- **Filled content in** ``add_new_attacks.rst``: Added a guide on how to implement new backdoor attacks, using an example similar to the image-based ``BadNet`` attack.
  

2024-10-18
-----------
- **Updated titles in** ``tasks.rst``: Revised section titles for improved readability.
- **Modified** ``add_new_models.rst`` **and** ``contributing.rst``: Made formatting adjustments to ensure consistency in ``.rst`` structure.
  

2024-10-19
-----------
- **Filled content in** ``add_new_datasets.rst``: Provided instructions on adding new datasets to BackdoorMBTI, with CIFAR-100 as an example.


2024-10-20
-----------
- **Filled content in** ``add_new_defenses.rst``: Added a guide on implementing new defense methods, using ``STRIP`` as an example for detecting poisoned data.


2024-10-22
-----------
- **Updated** ``add_new_defenses.rst``: Corrected ``.rst`` formatting issues for consistency.


2024-10-23
-----------
- **Filled content in** ``models.rst``: Provided an overview of models used in the project, categorized by task type.
- **Modified** ``attacks.rst``: Adjusted content for improved organization and readability.


2024-10-24
-----------
- **Modified** ``attacks.rst``: Made additional content adjustments for clarity.


2024-10-26
-----------
- **Filled content in** ``defenses.rst``: Added a table listing the supported defenses in BackdoorMBTI.


2024-11-07
-----------
- **Filled content in** ``poison_dataset_wrapper.rst``: This wrapper allows users to inject specific labels into a dataset at a controlled rate, making it especially useful for creating backdoor attacks in machine learning datasets.
- **Filled content in** ``configs.rst``: The ``Configs`` directory contains various configuration files and settings used for attacks, defenses, and training in the backdoor learning system.