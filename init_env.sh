##########################
# init conda environment #
##########################

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda create --name conda_env -y python=3.8
conda activate conda_env
pip install wandb
pip install torch==1.12.1
pip install matplotlib
pip install pandas
pip install transformers
pip install scipy
pip install scikit-learn
wandb login