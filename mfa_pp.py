# Train a Bayesian mixture of factor analysis with EM
# Most of this comes from Richardson and Weiss's torch-MFA implementation
import PIL.ImageChops
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import math
import warnings


class MFA(torch.nn.Module):
    def __init__(self, n_components, n_features, n_factors, alpha, isotropic_noise=True, init_method='rnd_samples'):
        super(MFA, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_factors = n_factors
        self.init_method = init_method
        self.isotropic_noise = isotropic_noise

        # Here are the hyperparameters for defining the priors
        self.alpha = alpha

        self.MU = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.A = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False)

        # Instead of optimizing for D, let's keep it fixed.
        self.log_D = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        # self.log_D = torch.nn.Parameter(torch.ones(n_components, n_features), requires_grad=False) * 0.02

        self.PI_logits = torch.nn.Parameter(torch.log(torch.ones(n_components)/float(n_components)), requires_grad=False)

    def sample(self, n, with_noise=False):
        """
        Generate random samples from the trained MFA / MPPCA
        :param n: How many samples
        :param with_noise: Add the isotropic / diagonal noise to the generated samples
        :return: samples [n, n_features], c_nums - originating component numbers
        """
        if torch.all(self.A == 0.):
            warnings.warn('SGD MFA training requires initialization. Please call batch_fit() first.')

        K, d, l = self.A.shape
        c_nums = np.random.choice(K, n, p=torch.softmax(self.PI_logits, dim=0).detach().cpu().numpy())
        z_l = torch.randn(n, l, device=self.A.device)
        z_d = torch.randn(n, d, device=self.A.device) if with_noise else torch.zeros(n, d, device=self.A.device)
        samples = torch.stack([self.A[c_nums[i]] @ z_l[i] + self.MU[c_nums[i]] + z_d[i] * torch.exp(0.5*self.log_D[c_nums[i]])
                               for i in range(n)])
        return samples, c_nums

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        # U, S, V = torch.svd(x - mu.reshape(1, -1))    # torch svd is less memory-efficient
        U, S, V = np.linalg.svd((x - mu.reshape(1, -1)).cpu().numpy(), full_matrices=False)
        V = torch.from_numpy(V.T).to(x.device)
        S = torch.from_numpy(S).to(x.device)
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0)) / ((x.shape[0] - 1) * (x.shape[1] - n_factors))
        A = V[:, :n_factors] * torch.sqrt(
            (torch.pow(S[:n_factors], 2.0).reshape(1, n_factors) / (x.shape[0] - 1) - sigma_squared))
        # return mu, A, torch.log(sigma_squared) * torch.ones(x.shape[1], device=x.device)
        return mu, A, torch.log(0.02*torch.ones(x.shape[1], device=x.device))

    def _init_from_data(self, x, samples_per_component, feature_sampling=False, init_PI_with_kmeans=False):
        from sklearn.decomposition import FactorAnalysis
        n = x.shape[0]
        K, d, l = self.A.shape

        if self.init_method == 'kmeans':
            # Import this only if 'kmeans' method was selected (not sure this is a good practice...)
            from sklearn.cluster import KMeans
            sampled_features = np.random.choice(d, int(d * feature_sampling)) if feature_sampling else np.arange(d)

            t = time.time()
            print('Performing K-means clustering of {} samples in dimension {} to {} clusters...'.format(
                x.shape[0], sampled_features.size, K))
            _x = x[:, sampled_features].cpu().numpy()
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(_x)

            if init_PI_with_kmeans:
                self.PI_logits.data = torch.log(torch.from_numpy(np.array([np.count_nonzero(clusters.labels_ == i)/n for i in range(K)]))).to(x.device)

            print('... took {} sec'.format(time.time() - t))
            component_samples = [clusters.labels_ == i for i in range(K)]
        elif self.init_method == 'rnd_samples':  # No clue what this means
            m = samples_per_component
            o = np.random.choice(n, m * K, replace=False) if m * K < n else np.arange(n)
            assert n >= m * K
            component_samples = [[o[i * m:(i + 1) * m]] for i in range(K)]

        # Still need to apply PCA to our data to create that low rank + diagonal structure
        params = [torch.stack(t) for t in zip(
            *[MFA._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.MU.data = params[0]
        self.A.data = params[1]
        # self.log_D = self.log_D.to(self.A.device())
        self.log_D.data = params[2]

    def _parameters_sanity_check(self):
        K, d, l = self.A.shape
        assert torch.all(torch.softmax(self.PI_logits, dim=0) > 0.01/K), self.PI_logits # The program expects component weights to have to be above some threshold. Why?
        assert torch.all(torch.exp(self.log_D) > 1e-5) and torch.all(torch.exp(self.log_D) < 1.0), \
            '{} - {}'.format(torch.min(self.log_D).item(), torch.max(self.log_D).item())
        assert torch.all(torch.abs(self.A) < 10.0), torch.max(torch.abs(self.A))
        assert torch.all(torch.abs(self.MU) < 1.0), torch.max(torch.abs(self.MU)) # The program expects all elements of the mean vector to be smaller than 1. Why is that?

    # This is where I incorporate the inverse wishart and dirchlet prior
    @staticmethod
    def _component_log_likelihood(x, PI_logits, MU, A, log_D, alpha):
        # A quick reminder for the dimensionality of these parameters
        # PI = K x 1
        # MU = K x d
        # A = K x d x l
        # log_D = K x d

        K, d, l = A.shape
        AT = A.transpose(1, 2)  # K x l x d
        iD = torch.exp(-log_D).view(K, d, 1)  # K x d x 1
        L = torch.eye(l, device=A.device).reshape(1, l, l) + AT @ (iD*A)  # K x l x l
        iL = torch.inverse(L)  # K x l x l

        def per_component_md(i):
            x_c = (x - MU[i].reshape(1, d)).T  # d x n
            m_d_1 = (iD[i] * x_c) - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] @ (iD[i] * x_c))
            return torch.sum(x_c * m_d_1, dim=0)  # 1 x n

        # Trying to implement the wishart prior -- never quite worked but that's okay! We just focus on dirichlet right now
        # def per_component_tr_precision_psi(i, psi):
        #     # define the precision matrix (equivalent to Sigma inverse)
        #     # iSigma = iD - iD @ A @ iL @ AT @ iD # This is wrong but it's the baseline
        #     # iD[i] = d x 1
        #     # iD[i] * A[i] = (d x 1) * (d x l) = d x l
        #     # (iD[i] * A[i]) @ iL[i] = (d x l) * (l x l) = d x l
        #     # ((iD[i] * A[i]) @ iL[i]) @ (AT[i] * iD[i]) = (d x l) * (l x d) = d x d
        #     precision = iD[i] - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] * iD[i].view(1, -1))  # d x d
        #     return torch.trace(precision @ psi)
        # tr_precision_psi = torch.stack([per_component_tr_precision_psi(i, psi) for i in range(K)])
        # log_inverse_wishart = -0.5*((nu+d+1)*log_det_Sigma + tr_precision_psi).view(K, 1)

        m_d = torch.stack([per_component_md(i) for i in range(K)])  # K x n
        det_L = torch.logdet(L)
        log_det_Sigma = det_L - torch.sum(torch.log(iD.reshape(K, d)), axis=1)  # K x 1
        log_prob_data_given_components = -0.5 * ((d*np.log(2.0*math.pi) + log_det_Sigma).reshape(K, 1) + m_d)
        log_dirichlet = torch.sum((alpha-1) * PI_logits)

        return PI_logits.reshape(1, K) + log_prob_data_given_components.T + log_dirichlet

    def per_component_log_likelihood(self, x, sampled_features=None):

        # Richardson and Weiss implemented this but I believe it's incorrect
        # if sampled_features is not None:
        #     return MFA._component_log_likelihood(x[:, sampled_features], torch.softmax(self.PI_logits, dim=0),
        #                                          self.MU[:, sampled_features],
        #                                          self.A[:, sampled_features],
        #                                          self.log_D[:, sampled_features], self.alpha)
        # return MFA._component_log_likelihood(x, torch.softmax(self.PI_logits, dim=0), self.MU, self.A, self.log_D, self.alpha)

        if sampled_features is not None:
            return MFA._component_log_likelihood(x[:, sampled_features], self.PI_logits,
                                                 self.MU[:, sampled_features],
                                                 self.A[:, sampled_features],
                                                 self.log_D[:, sampled_features],
                                                 self.alpha)
        return MFA._component_log_likelihood(x, self.PI_logits, self.MU, self.A, self.log_D, self.alpha)

    # compute log probability for one sample
    def log_prob(self, x, sampled_features=None):
        return torch.logsumexp(self.per_component_log_likelihood(x, sampled_features), dim=1)

    # sum log probability across all samples
    def log_likelihood(self, x, sampled_features=None):
        return torch.sum(self.log_prob(x, sampled_features))

    def log_responsibilities(self, x, sampled_features=None):
        """
        Calculate the log-responsibilities (log of the responsibility values - probability of each sample to originate
        from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-responsibilities values [n, n_components]
        """
        comp_LLs = self.per_component_log_likelihood(x, sampled_features)
        return comp_LLs - torch.logsumexp(comp_LLs, dim=1).reshape(-1, 1)

    def responsibilities(self, x, sampled_features=None):
        """
        Calculate the responsibilities - probability of each sample to originate from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: responsibility values [n, n_components]
        """
        return torch.exp(self.log_responsibilities(x, sampled_features))

    def batch_fit(self, train_dataset, test_dataset=None, batch_size=1000, test_size=1000, max_iterations=20,
                  feature_sampling=False, init_PI_with_kmeans=False):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        Memory-efficient batched implementation for large datasets that do not fit in memory:
        E step:
            For all mini-batches:
            - Calculate and store responsibilities
            - Accumulate sufficient statistics
        M step:
            Re-calculate all parameters
        Note that incremental EM per Neal & Hinton, 1998 is not supported, since we can't maintain
            the full x x^T as sufficient statistic - we need to multiply by A to get a more compact
            representation.
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_iterations: number of iterations (=epochs)
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
       """
        assert self.isotropic_noise, 'EM fitting is currently supported for isotropic noise (MPPCA) only'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.A.shape

        init_samples_per_component = (l+1)*2 if self.init_method == 'rnd_samples' else (l+1)*10
        print('Random init using {} with {} samples per component...'.format(self.init_method, init_samples_per_component))
        init_keys = [key for i, key in enumerate(RandomSampler(train_dataset)) if i < init_samples_per_component*K]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self._init_from_data(torch.stack(init_samples).to(self.MU.device),
                             samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling/2 if feature_sampling else False,
                             init_PI_with_kmeans=init_PI_with_kmeans)

        # Read some test samples for test likelihood calculation
        # test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.MU.device)

        ll_log = []
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        for it in range(max_iterations):
            t = time.time()

            # Sufficient statistics
            sum_r = torch.zeros(size=[K], dtype=torch.float64, device=self.MU.device)
            sum_r_x = torch.zeros(size=[K, d], dtype=torch.float64, device=self.MU.device)
            sum_r_x_x_A = torch.zeros(size=[K, d, l], dtype=torch.float64, device=self.MU.device)
            sum_r_norm_x = torch.zeros(K, dtype=torch.float64, device=self.MU.device)

            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('Iteration {}/{}, log-likelihood={}:'.format(it, max_iterations, ll_log[-1]))

            Z = 0
            for batch_x, _ in loader:
                print('E', end='', flush=True)
                batch_x = batch_x.to(self.MU.device)
                sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
                batch_r = self.responsibilities(batch_x, sampled_features=sampled_features)
                sum_r += torch.sum(batch_r, dim=0).double()

                # print(batch_x.shape)
                # print(batch_r.shape)
                # print(sum_r.shape)
                # print(torch.sum(sum_r))
                # print(torch.sum(batch_r, dim=1))

                sum_r_norm_x += torch.sum(batch_r * torch.sum(torch.pow(batch_x, 2.0), dim=1, keepdim=True), dim=0).double()
                # Z += torch.sum((batch_size + torch.sum(self.alpha) - K) / batch_size)
                for i in range(K):
                    batch_r_x = batch_r[:, [i]] * batch_x
                    sum_r_x[i] += torch.sum(batch_r_x, dim=0).double()
                    sum_r_x_x_A[i] += (batch_r_x.T.double() @ (batch_x.double() @ self.A[i].double())).double()

            print(' / M...', end='', flush=True)

            # MLE; equivalent to alpha = 1
            # self.PI_logits.data = torch.log(sum_r / torch.sum(sum_r)).float()

            Z = torch.sum(sum_r) + torch.sum(self.alpha) - self.n_components

            # printing out sum_r to see how that compares to self.alpha. We want to make sure sum_r isn't overwhelming
            # alpha if we want to see an effect of the dirchilet prior
            print(f'sum_r={sum_r}')
            print(f'alpha-1={self.alpha-1}')
            self.PI_logits.data = torch.log((sum_r + self.alpha-1) / Z)

            self.MU.data = (sum_r_x / sum_r.reshape(-1, 1)).float()
            SA = sum_r_x_x_A / sum_r.reshape(-1, 1, 1) - \
                 (self.MU.reshape(K, d, 1) @ (self.MU.reshape(K, 1, d) @ self.A)).double()
            s2_I = torch.exp(self.log_D[:, 0]).reshape(K, 1, 1) * torch.eye(l, device=self.MU.device).reshape(1, l, l)  # (K, l, l)
            M = (self.A.transpose(1, 2) @ self.A + s2_I).double()  # (K, l, l)
            inv_M = torch.stack([torch.inverse(M[i]) for i in range(K)])   # (K, l, l)
            invM_AT_S_A = inv_M @ self.A.double().transpose(1, 2) @ SA   # (K, l, l)
            self.A.data = torch.stack([(SA[i] @ torch.inverse(s2_I[i].double() + invM_AT_S_A[i])).float()
                                       for i in range(K)])
            t1 = torch.stack([torch.trace(self.A[i].double().T @ (SA[i] @ inv_M[i])) for i in range(K)])
            t_s = sum_r_norm_x / sum_r - torch.sum(torch.pow(self.MU, 2.0), dim=1).double()
            # self.log_D.data = torch.log((t_s - t1)/d).float().reshape(-1, 1) * torch.ones_like(self.log_D)

            self._parameters_sanity_check()
            print(' ({} sec)'.format(time.time()-t))

        ll_log.append(torch.mean(self.log_prob(test_samples)).item())
        print('\nFinal train log-likelihood={}:'.format(ll_log[-1]))
        return ll_log

    # We want to have a version that run the E-step and M-step for MU and PI but SGD for A and D
    def batch_hybrid_fit(self, train_dataset, test_dataset=None, batch_size=1000, test_size=1000, max_iterations=20,
                  feature_sampling=False, init_PI_with_kmeans=False, n_inner=5, learning_rate=0.0001):
        assert self.isotropic_noise, 'EM fitting is currently supported for isotropic noise (MPPCA) only'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.A.shape

        init_samples_per_component = (l + 1) * 2 if self.init_method == 'rnd_samples' else (l + 1) * 10
        print('Random init using {} with {} samples per component...'.format(self.init_method,
                                                                             init_samples_per_component))
        init_keys = [key for i, key in enumerate(RandomSampler(train_dataset)) if i < init_samples_per_component * K]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self._init_from_data(torch.stack(init_samples).to(self.MU.device),
                             samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling / 2 if feature_sampling else False,
                             init_PI_with_kmeans=init_PI_with_kmeans)

        # Read some test samples for test likelihood calculation
        # test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.MU.device)

        ll_log = []
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for it in range(max_iterations):
            self.PI_logits.requires_grad = self.MU.requires_grad = False
            self.A.requires_grad = self.log_D.requires_grad = False

            t = time.time()

            # Sufficient statistics
            sum_r = torch.zeros(size=[K], dtype=torch.float64, device=self.MU.device)
            sum_r_x = torch.zeros(size=[K, d], dtype=torch.float64, device=self.MU.device)
            sum_r_norm_x = torch.zeros(K, dtype=torch.float64, device=self.MU.device)

            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('Iteration {}/{}, log-likelihood={}:'.format(it, max_iterations, ll_log[-1]))

            Z = 0
            for batch_x, _ in loader:
                print('E', end='', flush=True)
                batch_x = batch_x.to(self.MU.device)
                sampled_features = np.random.choice(d, int(d * feature_sampling)) if feature_sampling else None
                batch_r = self.responsibilities(batch_x, sampled_features=sampled_features)
                sum_r += torch.sum(batch_r, dim=0).double()
                sum_r_norm_x += torch.sum(batch_r * torch.sum(torch.pow(batch_x, 2.0), dim=1, keepdim=True),
                                          dim=0).double()
                for i in range(K):
                    batch_r_x = batch_r[:, [i]] * batch_x
                    sum_r_x[i] += torch.sum(batch_r_x, dim=0).double()

            print(' / M...', flush=True)

            Z = torch.sum(sum_r) + torch.sum(self.alpha) - self.n_components
            self.PI_logits.data = torch.log((sum_r + self.alpha - 1) / Z)
            self.MU.data = (sum_r_x / sum_r.reshape(-1, 1)).float()

            self.A.requires_grad = True
            self.log_D.requires_grad = False
            for inner_iter in range(n_inner):
                for batch_x, _ in loader:
                    batch_x = batch_x.to(self.MU.device)
                    optimizer.zero_grad()
                    sampled_features = np.random.choice(d, int(d * feature_sampling)) if feature_sampling else None
                    loss = -self.log_likelihood(batch_x, sampled_features=sampled_features) / batch_size
                    loss.backward()
                    optimizer.step()
                print(f'Inner iteration {inner_iter}/{n_inner}, log-likelihood={self.log_prob(test_samples).mean().item()}')

            # self._parameters_sanity_check()
            print(' ({} sec)'.format(time.time() - t))

        ll_log.append(torch.mean(self.log_prob(test_samples)).item())
        print('\nFinal train log-likelihood={}:'.format(ll_log[-1]))
        self.PI_logits.requires_grad = self.MU.requires_grad = self.A.requires_grad = self.log_D.requires_grad = False
        return ll_log


    # Simplified version of batch_fit. Simply to call kmeans for initializing
    def init_from_data(self, train_dataset, test_dataset=None, batch_size=1000, test_size=1000, max_iterations=20, maximization_iterations=20,
                  feature_sampling=False, init_PI_with_kmeans=False):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        Memory-efficient batched implementation for large datasets that do not fit in memory:
        E step:
            For all mini-batches:
            - Calculate and store responsibilities
            - Accumulate sufficient statistics
        M step:
            Re-calculate all parameters
        Note that incremental EM per Neal & Hinton, 1998 is not supported, since we can't maintain
            the full x x^T as sufficient statistic - we need to multiply by A to get a more compact
            representation.
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_iterations: number of iterations (=epochs)
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
       """
        assert self.isotropic_noise, 'EM fitting is currently supported for isotropic noise (MPPCA) only'
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'
        K, d, l = self.A.shape

        init_samples_per_component = (l + 1) * 2 if self.init_method == 'rnd_samples' else (l + 1) * 10
        print('Random init using {} with {} samples per component...'.format(self.init_method,
                                                                             init_samples_per_component))
        init_keys = [key for i, key in enumerate(RandomSampler(train_dataset)) if i < init_samples_per_component * K]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self._init_from_data(torch.stack(init_samples).to(self.MU.device),
                             samples_per_component=init_samples_per_component,
                             feature_sampling=feature_sampling / 2 if feature_sampling else False,
                             init_PI_with_kmeans=init_PI_with_kmeans)

    def sgd_mfa_train(self, train_dataset, test_dataset=None, batch_size=128, test_size=1000, max_epochs=10,
                      learning_rate=0.0001, feature_sampling=False):
        """
        Stochastic Gradient Descent training of MFA (after initialization using MPPCA EM)
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_epochs: number of epochs
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
        """
        if torch.all(self.A == 0.):
            warnings.warn('SGD MFA training requires initialization. Please call batch_fit() first.')
        if self.isotropic_noise:
            warnings.warn('Currently, SGD training uses diagonal (non-isotropic) noise covariance i.e. MFA and not MPPCA')
        assert not feature_sampling or type(feature_sampling) == float, 'set to desired sampling ratio'

        self.PI_logits.requires_grad = self.MU.requires_grad = self.A.requires_grad = True
        self.log_D.requires_grad = False

        K, d, l = self.A.shape

        # Read some test samples for test likelihood calculation
        # test_samples, _ = zip(*[test_dataset[key] for key in RandomSampler(test_dataset, num_samples=test_size, replacement=True)])
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.MU.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        ll_log = []
        self.train()
        for epoch in range(max_epochs):
            t = time.time()
            for idx, (batch_x, _) in enumerate(loader):
                print('.', end='', flush=True)
                if idx > 0 and idx%100 == 0:
                    print(torch.mean(self.log_prob(test_samples)).item())
                sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else None
                batch_x = batch_x.to(self.MU.device)
                optimizer.zero_grad()
                loss = -self.log_likelihood(batch_x, sampled_features=sampled_features) / batch_size
                loss.backward()
                optimizer.step()
            ll_log.append(torch.mean(self.log_prob(test_samples)).item())
            print('\nEpoch {}: Test ll = {} ({} sec)'.format(epoch, ll_log[-1], time.time()-t))
            # self._parameters_sanity_check()
            print(torch.softmax(self.PI_logits, dim=0).detach().cpu().numpy())
        self.PI_logits.requires_grad = self.MU.requires_grad = self.A.requires_grad = self.log_D.requires_grad = False
        return ll_log
