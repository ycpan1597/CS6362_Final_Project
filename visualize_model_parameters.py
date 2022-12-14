import torch
import numpy as np
import matplotlib.pyplot as plt

model_path = '/home/pany11/Desktop/CS6362/torch-mfa-master/models/mnist/c_50_l_6_init_kmeans_sgd0_em100_initPI0_alpha5000.0_uniqueAlphaIndicesNone_MAP/model.pth'
model = torch.load(model_path)

plt.figure(1)
for i in range(model['MU'].shape[0]):
    plt.subplot(5, 10, i+1)
    plt.imshow(model['MU'][i, :].cpu().numpy().reshape(28, 28))
    plt.axis('off')
plt.show()

# Visualize covariance matrix
plt.figure(2)
for i in range(model['MU'].shape[0]):
    plt.subplot(5, 10, i+1)
    # plt.imshow(mdl1['MU'][i, :].cpu().numpy().reshape(28, 28))
    plt.imshow((model['A'][i] @ model['A'][i].T).cpu().numpy() + np.diag(torch.exp(model['log_D'][i]).cpu().numpy()))
    plt.axis('off')
plt.show()
