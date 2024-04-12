#!/bin/bash

python train.py --dataroot datasets/colon_fullres \
	--name colon_cyclegan_MIstruct_rgb_fullres_oblique \
	--model cycle_gan_struct \
	--lambda_struct 1.0 \
	--gpu_ids 0,1 --batch_size 4