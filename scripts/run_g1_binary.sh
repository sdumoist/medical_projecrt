#!/bin/bash
# Run G1 DenseNet binary classification

cd /mnt/cfs_algo_bj/models/experiments/lirunze/code/project

PYTHONPATH=. /root/miniforge3/envs/srre/bin/python train.py --config configs/g1_densenet_binary.yaml "$@"