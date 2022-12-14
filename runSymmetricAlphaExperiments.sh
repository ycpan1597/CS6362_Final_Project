#!/bin/bash

export PYTHONPATH=/home/pany11/Desktop/CS6362/torch-mfa-master

## run three symmetric alpha scalars for comparison (we use EM here because it's fastest)
# alpha = vector of all 2500
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 2500
# alpha = vector of all 5000
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 5000
# alpha = vector of all 10000
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 10000

## run alpha = 1 and alpha = 10000 for 10 components
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar -1 --n_components 10
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 10000 --n_components 10