#!/bin/bash

# Run training script on cluster
python prepare_dataset.py \
    --dataset_path /SAN/medic/RealCol/pilot_dataset \
    --synthetic1 /SAN/medic/RealCol/datasets_zip/synth2.zip \
    --synthetic2 /SAN/medic/RealCol/datasets_zip/synth3.zip \
    --real /SAN/medic/RealCol/datasets_zip/real.zip 
 