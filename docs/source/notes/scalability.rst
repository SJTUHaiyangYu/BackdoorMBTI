
Scalability
===========

BackdoorMBTI provides flexible scalability, allowing users to customize and extend the framework in the following aspects. Detailed guides are available to help with each type of addition:

1. **Adding New Datasets**  
   BackdoorMBTI supports loading custom datasets, allowing integration of any dataset (e.g., CIFAR-100 or custom datasets) into the framework. For details, please refer to the [Adding New Datasets Tutorial](../tutorials/add_new_datasets.rst).

2. **Adding New Attacks**  
   The framework allows users to implement new attack methods across various modalities (e.g., image, text, audio), such as the BadNet attack. For more information, see the [Adding New Attacks Tutorial](../tutorials/add_new_attacks.rst).

3. **Adding New Defenses**  
   BackdoorMBTI supports adding and testing new defense methods, such as STRIP, to enhance system robustness. See the [Adding New Defenses Tutorial](../tutorials/add_new_defenses.rst) for more details.

4. **Adding New Models**  
   The framework supports numerous models (e.g., ViT, GPT-2) and allows users to integrate custom model architectures, including PyTorch models and other pretrained models. For details, please refer to the [Adding New Models Tutorial](../tutorials/add_new_models.rst).

BackdoorMBTI is designed to ensure ease of expansion across these core modules. Each tutorial provides clear example code and instructions to assist users in quickly integrating and testing new functional components.
