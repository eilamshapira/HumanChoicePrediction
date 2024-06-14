# Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation


## Getting Started


### Prerequisites

Before you begin, ensure you have the following tools installed on your system:
- Git
- Anaconda or Miniconda

### Installation

To install and run the code on your local machine, follow these steps:

1. **Clone the repository**

   First, clone the repository to your local machine using Git. Open a terminal and run the following command:
   ```bash
   git clone https://github.com/eilamshapira/HumanChoicePrediction
    ```
2. **Create and activate the conda environment**

    After cloning the repository, navigate into the project directory:

    ```bash
    cd HumanChoicePrediction
    ```

    Then, use the following command to create a conda environment from the requirements.yml file provided in the project:
    ```bash
    conda env create -f requirements.yml
    ```
3. **Log in to Weights & Biases (W&B)**

   Weights & Biases is a machine learning platform that helps you track your experiments, visualize data, and share your findings. Logging in to W&B is essential for tracking the experiments in this project. If you haven't already, you'll need to create a W&B account. 
   Use the following command to log in to your account:
    ```bash
    wandb login
    ```

## Citation

If you find this work useful, please cite our paper:

    @misc{shapira2024human,
          title={Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation}, 
          author={Eilam Shapira and Reut Apel and Moshe Tennenholtz and Roi Reichart},
          year={2024},
          eprint={2305.10361},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

#hwlloooo