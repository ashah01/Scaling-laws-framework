#!/bin/bash

# Define hyperparameter values to search over
depths=(2 3 4)

# Loop over all possible combinations of hyperparameter values


for depth in "${depths[@]}"
do
  # Run your training script with the current hyperparameters
  python train.py \
  --batch_size 64 \
  --lr 0.0003 \
  --hidden_dim 40 \
  --depth $depth \
  --dropout1 0.15 \
  --dropout2 0.1
done

