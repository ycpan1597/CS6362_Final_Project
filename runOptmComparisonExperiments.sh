#!/bin/bash

export PYTHONPATH=/home/pany11/Desktop/CS6362/torch-mfa-master

## run MLE with three versions of optimization. Setting alpha_scalr to -1 = setting everything to MLE
# EM
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar -1
# EM-SGD
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 5 --init_pi_with_kmeans 0 --alpha_scalar -1
# SGD
python train_pp.py --mfa_sgd_epochs 50 --mfa_em_epochs 0 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar -1
