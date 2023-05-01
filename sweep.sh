#!/bin/bash

# Define hyperparameter values to search over
embedding_dims=(30)
depths=(2 3 4)

# Loop over all possible combinations of hyperparameter values

for embedding_dim in "${embedding_dims[@]}"
do
  for depth in "${depths[@]}"
  do
    # Run your training script with the current hyperparameters
    python train.py \
    --batch_size 32 \
    --lr 0.0003 \
    --hidden_dim $embedding_dim \
    --depth $depth \
    --dropout1 0.1 \
    --dropout2 0.1
  done
done

