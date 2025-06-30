.. _install:

Install
=======

To install BackdoorMBTI, please follow these steps:

1. **Install necessary system libraries**:

   Before setting up your environment, make sure to install necessary system libraries (we test it on Ubuntu 22.04). Run the following commands:

   .. code-block:: bash
      
      sudo apt-get update
      sudo apt-get install libgl1-mesa-glx

2. **Set up a Python virtual environment with conda**:

   After installing the necessary system libraries, create and activate a `conda` environment with Python 3.10:

   .. code-block:: bash

      conda create -n bkdmbti python=3.10
      conda activate bkdmbti

3. **Install from PyPI**:

   Once the environment is set up, run the following command to install BackdoorMBTI from PyPI:

   .. code-block:: bash

      pip install backdoormbti

4. **Install from source** (Recommanded):

   Alternatively, if you prefer to install from source, follow these steps:

   .. code-block:: bash

      git clone https://github.com/SJTUHaiyangYu/BackdoorMBTI
      cd BackdoorMBTI
      pip install -r requirements.txt


Download Data
-------------

Download the necessary data if it cannot be automatically downloaded. Some data download scripts are provided in the `scripts` folder. Check the datasets suppported automatically download at section :ref:`supported-tasks`. 