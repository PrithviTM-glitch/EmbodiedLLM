## Guidelines to Create Minimal Code Notebooks

When creating minimal code notebook for benchmarking or fine tuning purposes, please follow these guidelines to ensure cleanliness, consistency and reproducibility of code.

The following are the guidelines and recommendations for creating minimal code notebooks:

1. **Specify Environment**:
   - Clearly understand the nature of the notebook and where it will be executed (e.g. local machine, cloud service like Google Colab, etc.).
    - Specify the required environment (e.g. Python version, libraries, etc.) in the notebook or in an accompanying README file.

2. **Use conda or virtual environments**:
   - Recommend using conda or virtual environments to manage dependencies and avoid conflicts with other projects. Specifically if cloning multiple repositories which may have confilicting dependencies.
   - Provide instructions on how to set up the environment if necessary.

3. **Install Dependencies**:
    - List all required libraries and their versions, or if making use of cloned repositories, try minimizing own depedency installations.
    - Use package managers like `pip` or `conda` to install dependencies.
    - If using `pip`, consider providing a `requirements.txt` file or making use of an existing one in cloned repositories.

4. **Clone Necessary Repositories**:
    - If the notebook relies on code from other repositories, provide clear instructions on how to clone them.
    - Try to mainly clone repositories that are public and easily accessible. Try not to rely on private repositories.

5. **Code Writing**:
    - In most cases for either benchmarking or fine-tuning, code is alread procided in the cloned repositories. Try to not write new code unless absolutely necessary.
    - If new code is necessary, ensure it is well-documented and follows best practices for readability and maintainability.
    - Use comments to explain complex sections of code.


## Task Instructions

### Completed Tasks:
- [x] Benchmarking Evo-1 on:
    - VLABench
    - Meta-World
    - LIBERO

### TO DO:
- [ ] Fine-tuning Evo-1 on:
    - VLABench 
- [ ] Test Fine tunes Evo-1 on:
    - VLABench

### Fine tuning instructions:

Most of the code and examples for fine tuning Evo-1 on VLABench are already provided in the relevant repositories. 

VLABench: https://github.com/OpenMOSS/VLABench/tree/main
Evo-1: https://github.com/MINT-SJTU/Evo-1

Both repositories contain scripts and instructions on how to use their code for evaluation as well as fine tuning. 

The following pipeline is recommended for fine tuning the Evo-1 model on VLABench tasks:
1. Clone the VLABench repository and set up the environment as per their instructions.
2. Clone the Evo-1 repository and set up the environment as per their instructions.
3. Use the provided scripts in the VLABench repository to generate trajectories and convert them into the LeRobot v2.1 dataset format.
4. Use the Evo-1 training scripts to fine-tune the model on the generated datasets.
5. Update configuration files as necessary to point to the correct dataset locations and training parameters.
6. Evaluate the fine-tuned model using the evaluation scripts provided in the Evo-1 repository.

