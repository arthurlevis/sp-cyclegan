#!/bin/bash

python train.py \
	--dataroot ./pilot_dataset \
	--name pilot_experiment \
	--model cycle_gan_struct \
	--lambda_struct 1.0 \
	--gpu_ids -1 \
	--batch_size 4 \
	--n_epochs 1 \
	--n_epochs_decay 1 \
	--save_epoch_freq 1