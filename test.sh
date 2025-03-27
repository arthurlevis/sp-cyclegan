#!/bin/bash

python test.py \
	--dataroot ./pilot_dataset \
	--name pilot_experiment \
	--model cycle_gan_struct \
	--gpu_ids -1 \
	--batch_size 4 \
	--num_test 10