Contributing to BackdoorMBTI
============================

Thank you for your interest in contributing to BackdoorMBTI! We appreciate contributions of all types, including code, documentation, and ideas. This guide will help you get started and understand how to contribute effectively.

Getting Started
---------------

To contribute to BackdoorMBTI, follow these steps:

1. Fork the repository.
2. Clone the fork to your local machine.
3. Create a new branch for your contribution.
4. Make your changes, ensuring your work follows our guidelines.
5. Push your changes to your fork and submit a pull request.

You can find our repository here: https://anonymous.4open.science/r/BackdoorMBTI-D6A1/README.md
Setting up the Environment
---------------------------

1. Clone the repository:

   .. code-block:: bash

       git clone https://anonymous.4open.science/r/BackdoorMBTI-D6A1/README.md
       cd BackdoorMBTI

2. Create a virtual environment:

   .. code-block:: bash

       python3 -m venv venv
       source venv/bin/activate

3. Install dependencies:

   .. code-block:: bash

       pip install -r requirements.txt

Code Contributions
------------------

We follow a few coding standards to ensure that BackdoorMBTI maintains high quality and consistency. Please adhere to the following guidelines:

1. **PEP 8**: Ensure your code follows the PEP 8 Python coding standard.
2. **Unit Tests**: Write unit tests for any new functionality or changes to existing functionality.
3. **Type Annotations**: Use type annotations where possible to ensure better code clarity and maintainability.
4. **Linting**: Run the linter before submitting a pull request:

   .. code-block:: bash

       flake8

Documentation
-------------

If you are contributing documentation, we follow the reStructuredText (reST) format for our docs. Please ensure that you:

1. Follow the same structure and tone as existing documentation.
2. Use clear, concise language.
3. Add examples where applicable.

Adding or Updating Documentation
--------------------------------

To add or update documentation:

1. Edit the relevant `.rst` files in the `docs/` directory.
2. Build the documentation locally to ensure there are no formatting errors:

   .. code-block:: bash

       cd docs
       make html

Testing Your Changes
--------------------

We encourage you to run the test suite before submitting your contribution to ensure everything is functioning correctly.

1. Install test dependencies:

   .. code-block:: bash

       pip install -r test-requirements.txt

2. Run tests:

   .. code-block:: bash

       pytest

Submitting Your Contribution
-----------------------------

Once you have made your changes and verified that they meet the contribution guidelines:

1. **Commit your changes**: Make sure to write a meaningful commit message.
   
   .. code-block:: bash

       git commit -m "Description of your changes"

2. **Push your branch** to your fork:

   .. code-block:: bash

       git push origin your-branch-name

3. **Submit a pull request**: Go to the GitHub repository, navigate to the "Pull Requests" tab, and submit your PR.

We will review your contribution and provide feedback if necessary. If your pull request is accepted, it will be merged into the main codebase.

Questions
---------

If you have any questions, feel free to open an issue or reach out to the project maintainers directly.

