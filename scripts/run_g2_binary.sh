#!/bin/bash
# Run G2 ResNet binary classification

cd /mnt/cfs_algo_bj/models/experiments/lirunze/code/project

PYTHONPATH=. /root/miniforge3/envs/srre/bin/python train.py --config configs/g2_resnet_binary.yaml "$@"