#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=23:59:00
#SBATCH --job-name=BS
#SBATCH --partition=gpu
#SBATCH --mem=64Gb
#SBATCH --gres=gpu:k40m:1
#SBATCH --output=result_bs.%j.out

# module load cuda/9.2
# module load anaconda3/3.7

# cd /scratch/rezaei.b/BackgroundSubtraction/BackgroundSubtraction_LowRankVAE/code
# source activate torch_py3

# nvcc -V

# nvidia-smi

echo "--------------------------"
lr=0.0015
alpha=2000
epochs=1000
recon_path="../output/recon/BMC2012/Video_00$1"
ckpt_dir="../output/checkpoints/BMC2012/Video_00$1"
vid_path="/media/behnaz/My Book/BackgroundSubtraction/Data/BMC2012/Video_00$1/frames"

mkdir -p $recon_path
mkdir -p $ckpt_dir

python train_singleVid.py -lr $lr -epochs $epochs -clip 1 \
    -decay_step_list 50 100 150 200 250 300 350 400 450 500 \
    -vid_path "${vid_path}" --train_with_eval -alpha $alpha -ckpt_dir "${ckpt_dir}" -recon_path "${recon_path}/images" &>> "${recon_path}/log.txt"
