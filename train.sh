#!/bin/bash

#$ -N job_name
#$ -l tscratch=<n>G
#$ -l tmem=<n>G
#$ -l gpu=true
#$ -q gpu.q

# Create scratch directory
mkdir -p /scratch0/username/${JOB_ID}

# Copy training data to scratch
cp -r path/to/training_data /scratch0/username/${JOB_ID}/

# Cleanup function
function finish {
  rm -rf /scratch0/username/${JOB_ID}
}
trap finish EXIT ERR INT TERM

# Run training script on cluster
python train.py \ 
	--dataroot /scratch0/username/${JOB_ID}/data \ 
	--name experiment_name \ 
	--model cycle_gan_struct \ 
	--lambda_struct 1.0        # added

