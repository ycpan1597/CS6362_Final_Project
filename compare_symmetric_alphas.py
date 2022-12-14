import os
import torch
import numpy as np
import matplotlib.pyplot as plt

### Use this section to generate the comparison of component weights across the four models ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndicesNone_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha5000.0_uniqueAlphaIndicesNone_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndicesNone_MAP/model.pth']
labels = ['MLE', 'alpha=2500', 'alpha=5000', 'alpha=10000']
plt.figure()
for i in range(len(model_paths)):
    model_path = os.path.join(root_path, model_paths[i])
    model = torch.load(model_path)
    plt.plot(torch.softmax(model['PI_logits'], dim=0).cpu(), label=labels[i])
plt.legend()
plt.show()
plt.xlabel('Components')
plt.ylabel('Component weights')
### ---- ####

# ### Use this section to generate the comparison of component weights across the four models ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd50_em0_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndicesNone_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha5000.0_uniqueAlphaIndicesNone_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndicesNone_MAP/model.pth']
labels = ['MLE', 'alpha=2500', 'alpha=5000', 'alpha=10000']
plt.figure()
for i in range(len(model_paths)):
    model_path = os.path.join(root_path, model_paths[i])
    model = torch.load(model_path)
    plt.plot(torch.softmax(model['PI_logits'], dim=0).cpu(), label=labels[i])
plt.legend()
plt.show()
plt.xlabel('Components')
plt.ylabel('Component weights')
# ### ---- ####


#### Use this section to generate the component mean for the MLE and alpha=10000 model for 50 components; togger the index between 0 and 1 ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndicesNone_MAP/model.pth']
model = torch.load(os.path.join(root_path, model_paths[0]))
plt.figure(1)
for i in range(model['MU'].shape[0]):
    plt.subplot(5, 10, i+1)
    plt.imshow(model['MU'][i, :].cpu().numpy().reshape(28, 28))
    plt.axis('off')
plt.show()
#### ---- ####


#### Use this section to generate the component mean for the MLE and alpha=10000 model for 50 components; togger the index between 0 and 1 ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_10_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_10_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndicesNone_MAP/model.pth']
model = torch.load(os.path.join(root_path, model_paths[1]))
plt.figure(1)
for i in range(model['MU'].shape[0]):
    plt.subplot(1, 10, i+1)
    plt.imshow(model['MU'][i, :].cpu().numpy().reshape(28, 28))
    plt.axis('off')
plt.show()
#### ---- ####

# # Visualize covariance matrix
plt.figure(2)
for i in range(model['MU'].shape[0]):
    plt.subplot(5, 10, i+1)
    # plt.imshow(mdl1['MU'][i, :].cpu().numpy().reshape(28, 28))
    plt.imshow((model['A'][i] @ model['A'][i].T).cpu().numpy() + np.diag(torch.exp(model['log_D'][i]).cpu().numpy()))
    plt.axis('off')
plt.show()
