#!/bin/bash

# Run training script on cluster
python prepare_dataset.py \
    --dataset_path /SAN/medic/RealCol/exp1_dataset \
    --synthetic1 /SAN/medic/RealCol/datasets_zip/SyntheticColon_II.zip \
    --synthetic2 /SAN/medic/RealCol/datasets_zip/SyntheticColon_III.zip \
    --real /SAN/medic/RealCol/datasets_zip/oblique_train.zip 
 