source ~/anaconda3/etc/profile.d/conda.sh
conda activate conda_env

wandb_sweep="PASTE_SWEEPID_HERE"
num_runs_per_cuda=2
for ((j=0; j<num_runs_per_cuda; j++))
do
  num_gpu=$(nvidia-smi -L | wc -l)
  for ((i=0; i<num_gpu; i++))
  do
    export CUDA_VISIBLE_DEVICES=$i
    screen -dmS cuda{$i}agent{$j} wandb agent $wandb_sweep
  done
done
