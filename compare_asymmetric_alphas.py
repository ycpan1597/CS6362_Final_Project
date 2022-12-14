import os
import torch
import numpy as np
import matplotlib.pyplot as plt

### Use this section to generate the comparison of component weights across the models ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndices[0]_MAP/model.pth']
labels = ['MLE', 'first 5', 'first 10', 'first 25', 'first 1']
plt.figure()
for i in range(len(model_paths)):
    model_path = os.path.join(root_path, model_paths[i])
    model = torch.load(model_path)
    plt.plot(torch.softmax(model['PI_logits'], dim=0).cpu(), label=labels[i])
plt.legend()
plt.show()
plt.xlabel('Components')
plt.ylabel('Component weights')
#### ---- ####

#### Use this section to plot component mean ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndices[0]_MAP/model.pth']

for i in range(len(model_paths)):
    model = torch.load(os.path.join(root_path, model_paths[i]))
    plt.figure(1)
    for j in range(model['MU'].shape[0]):
        plt.subplot(5, 10, j+1)
        plt.imshow(model['MU'][j, :].cpu().numpy().reshape(28, 28))
        plt.axis('off')
    plt.show()
#### ---- ####

#### Use this section to plot component covariance of the first 5 component across the 5 models ####
root_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/'
model_paths = ['c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_MLE/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha2500.0_uniqueAlphaIndices[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_MAP/model.pth',
               'c_50_l_6_init_kmeans_sgd0_em50_hybridNinner0_initPI0_alpha10000.0_uniqueAlphaIndices[0]_MAP/model.pth']

cnt = 1
for i in range(len(model_paths)):
    model = torch.load(os.path.join(root_path, model_paths[i]))
    # for j in range(model['MU'].shape[0]):
    for j in range(5):
        plt.subplot(5, 5, cnt)
        # plt.imshow(mdl1['MU'][i, :].cpu().numpy().reshape(28, 28))
        plt.imshow((model['A'][j] @ model['A'][j].T).cpu().numpy() + np.diag(torch.exp(model['log_D'][j]).cpu().numpy()))
        plt.axis('off')
        cnt = cnt + 1
plt.show()
# ### ---- ####