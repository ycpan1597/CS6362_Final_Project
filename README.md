Here is my code for Vanderbilt CS6362 Final Project. Note that most of mfa_pp.py and train_pp.py are based on the 
original implementation by Richardson and Weiss in their 2018 NeruIPS paper  [On GANs and GMMs](https://arxiv.org/abs/1805.12462)

Please note that this is not their complete implementation as I only adapted part of it for this project. 
mfa.py and train.py are un-modified copies from Richardson and Weiss' github repository (https://github.com/eitanrich/torch-mfa)

Also, I used MNIST for this dataset. Train.py assumes the user has Pytorch installed on their machine and downloads MNIST through Pytorch. 

Here are the functions I modified:

In mfa_pp.py:
- batch_fit -- Modified the update of the PI_logits to incorporate the Dirichlet concentration parameter (alpha)
- batch_hybrid_fit -- This is identical to batch_fit in the E-step, but the M-step is slightly modified to run a few inner iterations of SGD to optimize the transformation matrix A
- _component_log_likelihood -- I added the dependence on alpha to this function
- _small_sample_PCA -- For the purpose of this project, I am not updating log_D; instead, I initialized the variance of the noise to be 0.02 across all features

In train_pp.py:
- Implemented a simple way to adjust the Dirichlet concentration vector

runOptmComparisonExperiments:
- Runs EM, EM-SGD, and SGD for optimizing MLE-GMM

runSymmetricAlphaExperiments:
- Runs multiple symmetric alpha vectors specified with the parameter "alpha_scalar"
- Results are visualized with compare_symmetric_alphas.py

runAsymmetricAlphaExperiments:
- Runs multiple asymmetric alpha vectors specified with the parameter "unique_alpha_indices"
- Results are visualized with compare_symmetric_alphas.py

