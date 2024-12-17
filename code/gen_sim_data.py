import numpy as np
import torch
from helper import *
import GPy

def gen_data_npr(N, d, img_shape, kernel='RBF', ls=0.5, var=1, sig=1, n_sim=1):
    '''
        generate data from nonparametric regression

        Args:
            N: number of observed data
            d: dimension of image data
            img_shape: image shape: [32, 32]
            kernel: true kernel function, 'RBF' or 'Matern32'
            ls: length scale of the kernel
            var: variance of the kernel 
            sig: noise standard deviation
            n_sim: number of data replications
        '''

    S = generate_grids(d, img_shape)
    V = S.shape[0]
    if kernel == 'RBF':
        k = GPy.kern.RBF(input_dim=d, lengthscale=ls, variance=var)
    elif kernel == 'Matern32':
        k = GPy.kern.Matern32(input_dim=d, lengthscale=ls, variance=var)
    else:
        print('kernel not valid')
    C = k.K(S.numpy(),S.numpy()) # covariance matrix
    mu = torch.zeros(V)
    sim_data = []
    for i in range(n_sim):
        np.random.seed(i)
        f = np.random.multivariate_normal(mu, C, size=N)  # True function samples (N x V)
        Y = f + np.random.normal(0, sig, (N, V))
        f = torch.from_numpy(f).t().float() # V x N
        Y = torch.from_numpy(Y).t().float() # V x N

        data = {'y':Y, 'f':f}
        sim_data.append(data)
    C = torch.from_numpy(C).float()
    res = {'kernel':C, 'grid':S, 'data':sim_data}
    return res

import numpy as np
import torch
from helper import *


def gen_inf_eff(N, S):
    '''
    generate N individual effect functions
    '''
    V = S.shape[0]
    eta = torch.zeros(V, N)

    # randomly select a location
    idx = torch.randint(V, (N,))
    # define the starting eff size and its sign
    prop_size = torch.FloatTensor(N).uniform_(0, 1)
    sign = torch.bernoulli(torch.ones(N)/2)
    # radial decrease or increase from the starting location
    for i in range(N):
        dist = (S[idx[i]] - S).pow(2).sum(1).sqrt() 
        dist = torch.exp(-dist)
        eta[:,i] = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
        eta[:,i] *= prop_size[i] * sign[i]
    return eta


def gen_data_IS_shapes(img_shape, N, beta, 
                        indiv_eff = True,
                        sigma_eps = 1, 
                        intercept = 1,
                        S = None, V_range = [-1,1],
                        n_sim = 1):
    '''
    generate simulation data given the spatial main effects
    
    Args:
            intercept: 0 or 1, indicate whether include the intercept term
    
    '''

    d = len(img_shape)
    assert(d == 2 or d == 3)
    assert(d == 2 or d == 3)
    
    V = beta.shape[0]
    J = beta.shape[1]
    assert(V == np.prod(img_shape))

    if S is None:
        generate_grids(d, V, img_shape, V_range)
    else:
        assert (V == S.shape[0])
        assert (d == S.shape[1])

    ### define selection indicator based on main effect shapes
    delta = torch.where(torch.isclose(beta, torch.tensor(0.0)), 0.0, 1.0)

    ### append the intercept term
    if intercept == 1:
        beta = torch.column_stack((torch.randn(V), beta))
        delta = torch.column_stack((torch.ones(V), delta))

    sim_data = []
    true_loglik = []
    for i in range(n_sim):
        torch.manual_seed(i)  ### set seeds
        
        X = torch.randn(N, J+intercept)
        if(intercept == 1):
            X[:, 0] = 1
        
        ### set indvidual effect
        if indiv_eff is True:
            eta = gen_inf_eff(N, S)
            eta = eta - eta @ (X @ torch.inverse(X.t() @ X) @ X.t()) ## standardize
        else: 
            eta = torch.zeros(V, N)       

        f = beta @ X.t() + eta
        eps = torch.randn(V, N) * sigma_eps ## noise
        y = f + eps

        ### save the true logliklihood
        y_dist = torch.distributions.Normal(f, sigma_eps)
        true_loglik.append(torch.sum(y_dist.log_prob(y)))

        data = {'y':y, 'X':X, 'f':f, 'eta':eta, 'loglik':true_loglik}
        sim_data.append(data)

    true_paras = {'beta':beta,
                  'delta':delta,
                  'sigma_eps':sigma_eps,
                  'grid':S}

    res = {'true_paras':true_paras, 'sim_data':sim_data}
    return res