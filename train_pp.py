import sys, os
import torch
import argparse
from torchvision.datasets import CelebA, MNIST
import torchvision.transforms as transforms
import numpy as np
from mfa_pp import MFA
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_model
from matplotlib import pyplot as plt
from imageio import imwrite
from packaging import version

"""
MFA model training (data fitting) example.
Note that actual EM (and SGD) training code are part of the MFA class itself.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mfa_sgd_epochs', default=100, type=int)
    parser.add_argument('--mfa_em_epochs', default=30, type=int)
    parser.add_argument('--mfa_hybrid_ninner', default=5, type=int)
    parser.add_argument('--init_pi_with_kmeans', default=0, type=int)
    parser.add_argument('--alpha_scalar', default=1, type=float, help='Setting this to -1 will result in the MLE')
    parser.add_argument('--unique_alpha_indices', nargs='*', type=int,
                        help='Should be a list of integers specifying which entries to have args.alpha_scalar. All '
                             'other entries will be 1')
    parser.add_argument('--n_components', default=50, type=int)
    args = parser.parse_args()

    assert version.parse(torch.__version__) >= version.parse('1.2.0')

    dataset = 'mnist'
    image_shape = [28, 28]  # The input image shape
    n_components = args.n_components  # Number of components in the mixture model (return this to 50 for experiments)
    n_factors = 6  # Number of factors - the latent dimension (same for all components)
    batch_size = 1000  # The EM batch size (return this to 1000 for experiments)
    feature_sampling = False  # For faster responsibilities calculation, randomly sample the coordinates (or False)
    init_method = 'kmeans'  # Initialize by using k-means clustering
    trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])
    train_set = MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = MNIST(root='./data', train=False, transform=trans, download=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = './models/' + dataset
    os.makedirs(model_dir, exist_ok=True)
    figures_dir = './figures/' + dataset
    os.makedirs(figures_dir, exist_ok=True)

    # List out all the parameters that we can change. These all need to be labeled appropriately

    # n_components (should be fixed at 50 unless necessary; for example, if this is too large for debugger to show values)
    # n_factors (should be fixed at 6 unless necessary)
    # args.mfa_sgd_epochs
    # args.mfa_em_epochs
    # k-means initialized PI_logits or not

    # These parameters only apply to MAP estimates
    # args.alpha_scalar
    # args.unique_alpha_indices

    if args.alpha_scalar == -1:
        alpha = torch.ones(n_components, device='cuda')  # Vector of all ones corresponds to MLE
        MLE_string = 'c_{}_l_{}_init_{}_sgd{}_em{}_hybridNinner{}_initPI{}_MLE'
        model_name = MLE_string.format(n_components, n_factors, init_method, args.mfa_sgd_epochs, args.mfa_em_epochs,
                                       args.mfa_hybrid_ninner, args.init_pi_with_kmeans)
    else:
        if not args.unique_alpha_indices:  # all entries are the same (symmetric)
            alpha = args.alpha_scalar * torch.ones(n_components, device='cuda')
        else:
            assert max(
                args.unique_alpha_indices) <= n_components - 1, f'maximum value in args.unique_alpha_indices must be smaller than {n_components}-1'
            alpha = torch.ones(n_components, device='cuda')
            alpha[args.unique_alpha_indices] = args.alpha_scalar
        # map
        MAP_string = 'c_{}_l_{}_init_{}_sgd{}_em{}_hybridNinner{}_initPI{}_alpha{}_uniqueAlphaIndices{}_MAP'
        model_name = MAP_string.format(n_components, n_factors, init_method, args.mfa_sgd_epochs, args.mfa_em_epochs,
                                       args.mfa_hybrid_ninner, args.init_pi_with_kmeans, args.alpha_scalar,
                                       args.unique_alpha_indices)
    print(alpha)

    this_model_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(this_model_dir):
        os.mkdir(this_model_dir)

    this_figure_dir = os.path.join(figures_dir, model_name)
    if not os.path.exists(this_figure_dir):
        os.mkdir(this_figure_dir)

    print('Defining the MFA model...')
    model = MFA(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors, alpha=alpha,
                init_method=init_method).to(device=device)

    print('EM fitting: {} components / {} factors / batch size {} ...'.format(n_components, n_factors, batch_size))

    if args.mfa_em_epochs > 0 and args.mfa_hybrid_ninner == 0:
        ll_log = model.batch_fit(train_set, test_set, batch_size=batch_size, max_iterations=args.mfa_em_epochs,
                                 feature_sampling=feature_sampling, init_PI_with_kmeans=args.init_pi_with_kmeans)

    if args.mfa_sgd_epochs > 0:
        model.init_from_data(train_set, init_PI_with_kmeans=args.init_pi_with_kmeans)
        print('Training using SGD with diagonal (instead of isotropic) noise covariance...')
        model.isotropic_noise = True
        ll_log = model.sgd_mfa_train(train_set, test_size=256, max_epochs=args.mfa_sgd_epochs,
                                     feature_sampling=feature_sampling)

    if args.mfa_hybrid_ninner > 0:
        print('Training using EM-SGD hybrid')
        model.isotropic_noise = True
        ll_log = model.batch_hybrid_fit(train_set, test_set, batch_size=batch_size,
                                        max_iterations=args.mfa_em_epochs, feature_sampling=feature_sampling,
                                        init_PI_with_kmeans=args.init_pi_with_kmeans, n_inner=args.mfa_hybrid_ninner)

    print('Saving the model...')
    # torch.save(model.state_dict(), os.path.join(model_dir, 'model_'+model_name+'.pth'))
    torch.save(model.state_dict(), os.path.join(this_model_dir, 'model.pth'))

    print('Visualizing the trained model...')
    model_image = visualize_model(model, image_shape=image_shape, end_component=10)
    # imwrite(os.path.join(figures_dir, 'model_'+model_name+'.jpg'), model_image)
    imwrite(os.path.join(this_figure_dir, 'model.jpg'), model_image)

    print('Generating random samples...')
    rnd_samples, _ = model.sample(100, with_noise=False)
    mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
    # imwrite(os.path.join(figures_dir, 'samples_'+model_name+'.jpg'), mosaic)
    imwrite(os.path.join(this_figure_dir, 'samples.jpg'), mosaic)

    print('Plotting test log-likelihood graph...')
    plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
    plt.grid(True)
    # plt.savefig(os.path.join(figures_dir, 'training_graph_'+model_name+'.jpg'))
    plt.savefig(os.path.join(this_figure_dir, 'training_graph.jpg'))
    print('Done')


if __name__ == "__main__":
    main()
