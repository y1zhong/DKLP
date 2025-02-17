import torch
import numpy as np
from torch import nn
import math
from tqdm import tqdm
from helper import *

class NPR():
    def __init__(self, S, Y, pred_S=None,
                J=50, H=50, M=4, act_fn='relu',
                ortho='GS', lr=0.1,
                burnin=500, thin=1, mcmc_sample=500,
                init_B_nn = None, 
                init_theta = None, 
                init_sigma2_eps = None,
                init_sigma2_nn = None, 
                init_lamb = None,
                init_a_lamb = None,
                diminishing_ratio= 0.1, r = 0.55, 
                a_nn=0.001, b_nn=0.001, a_eps=0.001, b_eps=0.001
                ):
        '''
        Initialize a nonparametric regression class trained by DKLP framework

        Args:
            S: V x D matrix for d-dimensional image grids on V locations
            Y: V x N matrix for N images data
            J: number of basis used to approximate kernel function 
            H (int): number of hidden units in each DNN layer
            M (int): number of hidden layer in DNN
            act_fn (str): activaton function in DNN, including ('relu','leaky','leaky0100','leaky0010','swish','sigmoid')
            ortho: the orthorgonlization operator, 'GS' or 'SVD'
            burnin (int): number of burnin 
            thin (int): number of thinning in mcmc samples
            mcmc_sample (int): number of mcmc samples for posterior inference
            lr: learning rate for SGLD algorithm
        '''
        self.y = Y
        self.S = S 
       
        self.V = S.shape[0] # num of locations
        self.d = S.shape[1] # grid dimension
       
        self.N = Y.shape[1]
        self.J = J 
        self.ortho = ortho 
        
        #mcmc settings
        self.mcmc_burnin = burnin
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample
        self.total_iter = burnin + mcmc_sample * thin

        #hyper parameters
        self.lr = lr 
        self.b_0 = self.total_iter/((1.0/diminishing_ratio)**(1.0/r) - 1.0)
        self.a_0 = lr*self.b_0**(r)
        self.r = r
        self.a_nn = a_nn
        self.b_nn = b_nn
        self.a_eps = a_eps
        self.b_eps = b_eps
        self.A2 = 100

        
        #initialization
        self.model = DKLP_NN(self.d, H, self.J, M=M, act_fn=act_fn)
        self.B_nn_cumsum_ind = np.cumsum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.B_nn_cumsum_ind = np.concatenate(([0], self.B_nn_cumsum_ind))        
        
        if init_B_nn is None:
            init_B_nn = torch.randn(self.B_nn_cumsum_ind[-1])

        if init_theta is None:
            init_theta = torch.randn(self.J, self.N)

        if init_sigma2_eps is None:
            init_sigma2_eps =  torch.tensor(1)

        if init_sigma2_nn is None:
            init_sigma2_nn = torch.tensor(1)

        if init_lamb is None:
            init_lamb = torch.sort(torch.rand(self.J), descending=True)[0]
        
        if init_a_lamb is None:
            init_a_lamb = torch.ones(self.J)

        self.B_nn =  init_B_nn.requires_grad_()
        self.theta = init_theta
        self.sigma2_eps = init_sigma2_eps
        self.sigma2_nn = init_sigma2_nn
        self.lamb = init_lamb
        self.a_lamb = init_a_lamb
        
        self.return_weights()
        self.nn_out = self.model(self.S)

        self.update_Psi_GS()
        self.f = self.Psi_detach @ self.theta
        self.update_y_tilda()

        
        self.mcmc_f = torch.zeros(self.mcmc_sample, self.V, self.N)
        #prediction on new spatial locations
        self.pred_S = pred_S
        if pred_S is not None:
            self.mcmc_pred_f = torch.zeros(pred_S.shape[0], self.N)
   
        #logliklihood
        self.log_lik_y = torch.zeros(self.total_iter) # save log liklihood

    def update_Psi_GS(self):
        self.Psi, R = torch.linalg.qr(self.nn_out)
        R_diag = R.diag()
        for l in range(self.J) :
            if(R_diag[l] < 0):
                self.Psi[:,l] = self.Psi[:,l] * (-1)
                R_diag[l] = R_diag[l] * (-1)
        self.Psi_detach = self.Psi.detach()
        self.update_f()

    def update_Psi_SVD(self):
        self.Psi, self.lambda_sqrt, _ = torch.linalg.svd(self.nn_out, full_matrices=False)
        for l in range(self.J) :
            if(self.Psi[0,l] < 0):
                self.Psi[:,l] = self.Psi[:,l] *(-1)
        self.Psi_detach = self.Psi.detach()
        self.lamb = self.lambda_sqrt.detach() ** 2
        self.update_f()

    def fit(self):
        if self.ortho == 'GS':
            self.fit_GS()
        elif self.ortho == 'SVD':
            self.fit_SVD()

    def fit_GS(self):
        for i in range(self.total_iter):
            self.lrt = self.a_0*(self.b_0 + i)**(-self.r) 
            self.epsilon = torch.randn(1) * math.sqrt(self.lrt)
            self.update_sigma2_eps()
            self.update_sigma2_nn()
            self.update_B_nn()
            self.update_Psi_GS()
            self.update_theta()
            self.update_lamb()
            self.update_a_lamb()
            
            self.update_loglik_y(i)
            if i >= self.mcmc_burnin:
                if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                    mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                    self.save_mcmc_samples(mcmc_iter)
                    if self.pred_S is not None:
                        self.pred(self.pred_S)
                    
    def fit_SVD(self):
        for i in range(self.total_iter):
            self.lrt = self.a_0*(self.b_0 + i)**(-self.r) 
            self.epsilon = torch.randn(1) * math.sqrt(self.lrt)
            self.update_sigma2_eps()
            self.update_sigma2_nn()
            self.update_B_nn()
            self.update_Psi_SVD()
            self.update_theta()
            self.update_loglik_y(i)
            if i >= self.mcmc_burnin:
                if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                    mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                    self.save_mcmc_samples(mcmc_iter)
                    if self.pred_S is not None:
                        self.pred(self.pred_S)
                  

    def update_y_tilda(self):
        self.y_tilda = self.Psi_detach.t() @ self.y

    def get_weights(self):
        param = torch.empty(0)
        for p in self.model.parameters():
            p_vec = p.reshape(-1)
            param = torch.cat((param, p_vec))
        return param

    def return_weights(self):
        for i, p in enumerate(self.model.parameters()):
            value = (self.B_nn[self.B_nn_cumsum_ind[i]:self.B_nn_cumsum_ind[i+1]]).clone().detach().requires_grad_(True).reshape(p.shape)
            p.data = value
    
    def update_lamb(self):
        a_eps_new = (1 + self.N ) / 2 
        b_eps_new = torch.sum(self.theta ** 2, 1) / 2 + 1 / self.a_lamb
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.lamb = 1 / m.sample()

    def update_a_lamb(self):
        b_eps_new = 1 / self.A2 + 1 / self.lamb
        m = torch.distributions.Gamma(1, b_eps_new)
        self.a_lamb = 1 / m.sample()

    def update_f(self):
        self.f = self.Psi_detach @ self.theta

    def update_theta(self):
        sigma2_theta = 1 / (1 / self.sigma2_eps + 1 / self.lamb.detach())[:,None] 
        mu_theta = ( self.y_tilda / self.sigma2_eps ) * sigma2_theta
        self.theta = torch.randn_like(self.theta) * sigma2_theta.sqrt() + mu_theta
        self.update_f()

    def update_B_nn(self):
        log_prior_nn = - 0.5 * torch.sum(self.B_nn ** 2) / self.sigma2_nn  
        y_res = self.Psi @ self.theta
        log_ll =  - 0.5 * torch.sum((self.y - y_res) ** 2) / self.sigma2_eps

        log_post = -(log_prior_nn +  log_ll)
        du_t = torch.autograd.grad(log_post/self.N, self.B_nn)[0]
        
        with torch.no_grad():
            self.B_nn +=  0.5 * self.lrt * du_t + self.epsilon
            self.B_nn.grad = None
        
        self.return_weights()
        self.nn_out = self.model(self.S)
            
    def update_sigma2_eps(self):
        a_eps_new = self.V * self.N / 2 + self.a_eps
        b_eps_new = torch.sum((self.y - self.f) ** 2) / 2 + self.b_eps
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.sigma2_eps = 1 / m.sample()
    
    def update_sigma2_nn(self):
        a_nn_new = self.B_nn_cumsum_ind[-1] / 2 + self.a_nn
        b_nn_new = torch.sum(self.B_nn.detach() ** 2) / 2 + self.b_nn
        m = torch.distributions.Gamma(a_nn_new, b_nn_new)
        self.sigma2_nn = 1 / m.sample()

    
    def update_loglik_y(self,i):
        self.log_lik_y[i] = -0.5 * self.N * torch.log(2 * torch.pi * self.sigma2_eps) - 0.5  * torch.sum((self.y - self.f) ** 2)/ self.sigma2_eps
    
    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_f[mcmc_iter,:,:] = self.f


    def post_summary(self):
        post_f = torch.median(self.mcmc_f, 0)[0]
        post_params = {'maineff': post_f}
       
        return  post_params

    def post_pred(self):
        post_pred_f = self.mcmc_pred_f/self.mcmc_sample
        return post_pred_f



    def pred(self, pred_S):
        nn_out = self.model(pred_S)
        Psi, R = torch.linalg.qr(nn_out)
        R_diag = R.diag()
        for l in range(self.J) :
            if(R_diag[l] < 0):
                Psi[:,l] = Psi[:,l] * (-1)
                R_diag[l] = R_diag[l] * (-1)
        Psi_detach = Psi.detach()
        f = Psi_detach @ self.theta
        self.mcmc_pred_f += f








