#!/bin/bash

python test.py \
	--dataroot ./pilot_dataset \
	--name pretrained \
	--epoch 30 \
	--model cycle_gan_struct \
	--gpu_ids -1 \
	--num_test 10