#!/bin/bash

#$ -N job_name
#$ -l tscratch=<n>G
#$ -l tmem=<n>G
#$ -l gpu=true
#$ -q gpu.q

# Create scratch directory
mkdir -p /scratch0/username/${JOB_ID}

# Copy test data to scratch
cp -r path/to/test_data /scratch0/username/${JOB_ID}/

# Cleanup function
function finish {
  rm -rf /scratch0/username/${JOB_ID}
}
trap finish EXIT ERR INT TERM

# Run test script on cluster
python test.py \
	--dataroot /scratch0/username/${JOB_ID}/data \