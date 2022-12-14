#!/bin/bash

export PYTHONPATH=/home/pany11/Desktop/CS6362/torch-mfa-master

## run three asymmetric alpha vectors
# alpha = first 5 are 2500, the remaining are 1
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 2500 --unique_alpha_indices 0 1 2 3 4
## alpha = first 10 are 2500, the remaining are 1
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 2500 --unique_alpha_indices 0 1 2 3 4 5 6 7 8 9
## alpha = first 25 are 2500, the remaining are 1
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 2500 --unique_alpha_indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
## alpha = first 1 is 10000, the remaining are 1
python train_pp.py --mfa_sgd_epochs 0 --mfa_em_epochs 50 --mfa_hybrid_ninner 0 --init_pi_with_kmeans 0 --alpha_scalar 10000 --unique_alpha_indices 0