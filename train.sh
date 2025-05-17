#!/bin/bash
#$ -N pilot_exp
#$ -l tscratch=4G
#$ -l tmem=16G
#$ -l gpu=true
#$ -q gpu.q

# Activate conda environment
source /home/alevisal/miniconda3/etc/profile.d/conda.sh
conda activate sp-cyclegan

# Create scratch directory
mkdir -p /scratch0/alevisal/${JOB_ID}

# Copy training data to scratch
cp -r /SAN/medic/RealCol/pilot_dataset /scratch0/alevisal/${JOB_ID}/

# Cleanup function
function finish {
  rm -rf /scratch0/alevisal/${JOB_ID}
}
trap finish EXIT ERR INT TERM

# Run 
cd /home/alevisal/structure-preserving-cyclegan

python train.py \
	--dataroot /scratch0/alevisal/${JOB_ID}/pilot_dataset \
	--name pilot_exp \
	--model cycle_gan_struct \
	--no_flip \
	--save_epoch_freq 1 \
	--n_epochs 2 \
	--n_epochs_decay 0 
