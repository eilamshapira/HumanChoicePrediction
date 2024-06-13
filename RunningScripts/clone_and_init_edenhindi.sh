# cloning your code from github:
git clone https://github.com/edenhindi/HumanChoicePrediction.git

cd HumanChoicePrediction

conda env create -f requirements.yml

conda activate final_project_env

wandb login
# Your main sweep:

python RunningScripts/final_sweep_edenhindi.py


# More runs appear in your report:
# python sweep_1.py
# python sweep_2.py
# python sweep_3.py
