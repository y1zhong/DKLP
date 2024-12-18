import torch
import numpy as np

import math
from tqdm import tqdm
from helper import log1pexp, DKLP_NN, FDR_SI
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, data
import rpy2.robjects.numpy2ri
ro.numpy2ri.activate()
pgdraw = importr('pgdraw')
pgdraw_f = rpy2.robjects.r['pgdraw']



class SI():
    def __init__(self, S, Y, X, U,
                J=50, H=128, M=4, act_fn='relu',
                ortho='GS',lr=0.01, ns = 1,
                first_burnin = 200, second_burnin=200,
                thin = 1, mcmc_sample = 500,
                batch_size = 100, 
                init_theta_nn=None,
                init_theta_f=None,
                init_theta_beta=None, 
                init_alpha=None,
                init_sigma2_nn=None, 
                init_sigma2_u=None,
                init_sigma2_f=None, 
                init_sigma2_alpha=None,
                init_theta_rho=None,
                init_delta = None,
                init_lamb=None,
                init_a_lamb=None,
                diminishing_ratio= 0.1, r = 0.55,
                a_f = 0.01, b_f = 0.01,
                a_u = 0.01, b_u = 0.01,
                a_nn = 0.01, b_nn = 0.01,
                a_alpha = 0.01, b_alpha = 0.01, A2 = 100,
                ):
        '''
        Initialize a scalar-on-image regression class trained by DKLP framework

        Args:
            S (torch.Tensor): V x D matrix for d-dimensional image grids on V locations
            Y (torch.Tensor): V x N matrix for N images data
            X (torch.Tensor): N x q matrix for q confounding covariates
            U (torch.Tensor): scalar vector for N response variables
            J (int): number of basis used to approximate kernel function 
            H (int): number of hidden units in each DNN layer
            ortho (str): the orthorgonlization operator, 'GS' or 'SVD'
            lr (float): learning rate for SGLD algorithm
            ns (float): normalizing scalar
            first_burnin (int): number of burnin without selection indicator
            second_burnin (int): number of burnin with selection indicator 
            thin (int): number of thinning in mcmc samples
            mcmc_sample (int): number of mcmc samples for posterior inference
            init_* (torch.Tensor): initial values for parameters
        '''

        self.y = Y 
        self.X = X 
        self.S = S
        self.V = S.shape[0] # num of locations
        self.d = S.shape[1] # grid dimension
        self.N = Y.shape[1] # num of individuals
        self.J = J
        self.U = U 
        self.ortho = ortho

        self.Xt = self.X.t()
        self.ssq_X = torch.sum(self.X ** 2, 0)
        self.q = self.X.shape[1]

        self.ns=ns
        
        # indices split for batch update
        self.n = min(batch_size, self.N)
        self.ind_split = torch.split(torch.arange(0, self.N), self.n)
        self.num_split = len(self.ind_split)

        # mcmc settings
        self.first_burnin = first_burnin
        self.second_burnin = second_burnin
        self.mcmc_burnin = first_burnin + second_burnin 
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample 
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning
        
        # hyperparameters
        self.lr = lr 
        self.b_0 = self.total_iter/((1.0/diminishing_ratio)**(1.0/r) - 1.0)
        self.a_0 = lr*self.b_0**(r)
        self.r = r

        self.a_f = a_f
        self.b_f = b_f
        self.a_u = a_u
        self.b_u = b_u
        self.a_nn = a_nn
        self.b_nn = b_nn
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.A2 = A2

        
        # initialization
        self.model = DKLP_NN(self.d, H, self.J, M=M, act_fn=act_fn)
        self.theta_nn_cumsum_ind = np.cumsum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.theta_nn_cumsum_ind = np.concatenate(([0], self.theta_nn_cumsum_ind))

        if init_theta_nn is None:
            init_theta_nn = torch.randn(self.theta_nn_cumsum_ind[-1])

        if init_theta_beta is None:
            init_theta_beta = torch.randn(self.J)
    
        if init_theta_f is None:
            init_theta_f =  torch.randn(self.J, self.N)

        if init_alpha is None:
            init_alpha = torch.randn(self.q)

        
        if init_delta is None:
            init_delta = torch.ones(self.V)

        if init_sigma2_f is None:
            init_sigma2_f =  torch.tensor(1)

        if init_sigma2_u is None:
            init_sigma2_u = torch.tensor(1)

        if init_sigma2_nn is None:
            init_sigma2_nn = torch.tensor(1)

        if init_sigma2_alpha is None:
            init_sigma2_alpha = torch.tensor(1)

        if init_theta_rho is None:
           init_theta_rho = torch.randn(self.J)
        
        if init_lamb is None:
           init_lamb = torch.ones(self.J)

        if init_a_lamb is None:
           init_a_lamb = torch.ones(self.J)

        self.theta_nn = init_theta_nn.requires_grad_()
        self.theta_f = init_theta_f
        self.theta_beta = init_theta_beta
        self.delta = init_delta
        self.alpha = init_alpha
        self.lamb = init_lamb
        self.a_lamb = init_a_lamb

        self.sigma2_f = init_sigma2_f
        self.sigma2_u = init_sigma2_u
        self.sigma2_nn = init_sigma2_nn
        self.sigma2_alpha = init_sigma2_alpha

       
        # intermediate parameters initialization
        self.return_weights()
        self.nn_out = self.model(self.S)
        if self.ortho == 'GS':
            self.update_Psi_GS(0)
        elif self.ortho == 'SVD':
            self.update_Psi_SVD()
        self.beta = self.Psi_detach @ self.theta_beta
        
        self.y_tilda = torch.zeros(self.J, self.N) # y transform, LxN
        self.f_est = torch.zeros(self.V, self.N) # est img functions, VxN
        self.u_est = torch.zeros(self.N) # est repsonse variable, N
        self.rss_y = 0 # residual sum of suqares
        self.resid_u = torch.zeros(self.N)
        self.rss_u = 0
       
        self.theta_rho = init_theta_rho
        
        self.set_mcmc_samples()
        self.set_loglik()
        
            
    def fit(self):
        for i in tqdm(range(self.total_iter)):
                self.lrt = self.a_0*(self.b_0 + i)**(-self.r) 
                self.eta = torch.randn(self.num_split) * math.sqrt(self.lrt)
                for k in range(self.num_split):
                    self.batch_ind = self.ind_split[k]
                    self.update_batch(k)

                    self.update_theta_nn(i,k) 
                    self.update_theta_beta(i) 
                    self.update_alpha(i)
                    
                    if(i >= self.first_burnin):
                        self.update_theta_rho(i)
                        self.update_delta()
                    
                    self.update_f_est()
                    self.update_u_est()
                    self.f_est[:, self.batch_ind] = self.f_est_batch
                    self.u_est[self.batch_ind] = self.u_est_batch


                    self.update_sigma2_u() 
                    self.update_sigma2_f() 
                    self.update_sigma2_alpha() 
                    self.update_sigma2_nn()

                    self.update_loglik_u(i)
                    self.update_loglik_y(i)
                if i >= self.mcmc_burnin:
                    if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                        mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                        self.save_mcmc_samples(mcmc_iter)

    def update_batch(self, k):
        self.y_batch = self.y[:, self.batch_ind]
        self.U_batch = self.U[self.batch_ind]
        self.X_batch = self.X[self.batch_ind]
        self.Xt_batch = self.Xt[:, self.batch_ind]
        self.ssq_X_batch = torch.sum(self.Xt_batch ** 2, 1)

        self.theta_f_batch = self.theta_f[:, self.batch_ind]
        self.f_est_batch = self.f_est[:, self.batch_ind]
        self.u_est_batch = self.u_est[self.batch_ind]

    def get_weights(self):
        param = torch.empty(0)
        for p in self.model.parameters():
            p_vec = p.reshape(-1)
            param = torch.cat((param, p_vec))
        return param

    def return_weights(self):
        for i, p in enumerate(self.model.parameters()):
            value = (self.theta_nn[self.theta_nn_cumsum_ind[i]:self.theta_nn_cumsum_ind[i+1]]).clone().detach().reshape(p.shape)
            p.data = value


    def update_f_est(self):
        self.f_est_batch = torch.mm(self.Psi_detach, self.theta_f_batch)
        self.rss_y = torch.sum((self.y_batch - self.f_est_batch) ** 2)
    
  
    def update_y_tilda(self):
        self.y_tilda = self.Psi_detach.t() @ self.y_batch

    def update_u_est(self):
        self.u_est_batch = self.y_batch.t() @ (self.beta * self.delta) / self.ns + self.X_batch @ self.alpha
        self.rss_u = torch.sum((self.U_batch - self.u_est_batch) ** 2)
    
    def update_Psi_GS(self,i):
        self.Psi, R = torch.linalg.qr(self.nn_out)
        R_diag = R.diag()
        for l in range(self.J) :
            if(R_diag[l] < 0):
                self.Psi[:,l] = self.Psi[:,l] * (-1)
                R_diag[l] = R_diag[l] * (-1)
        self.Psi_detach = self.Psi.detach()
        self.update_lamb(i)
        self.update_a_lamb()

    def update_Psi_SVD(self):
        self.Psi, self.lambda_sqrt, _ = torch.linalg.svd(self.nn_out, full_matrices=False)
        for l in range(self.J) :
            if(self.Psi[0,l] < 0):
                self.Psi[:,l] = self.Psi[:,l] *(-1)
        self.Psi_detach = self.Psi.detach()
        self.lamb = self.lambda_sqrt.detach() ** 2

    def update_lamb(self,i):
        a_eps_new = (1 + 1  ) / 2 
        b_eps_new = (self.theta_beta ** 2) / 2 + 1 / self.a_lamb
        if i >= self.first_burnin:
            a_eps_new += 1 / 2 
            b_eps_new += (self.theta_rho ** 2) / 2
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.lamb = 1 / m.sample()

    def update_a_lamb(self):
        b_eps_new = 1 / self.A2 + 1 / self.lamb
        m = torch.distributions.Gamma(1, b_eps_new)
        self.a_lamb = 1 / m.sample()

        
    
    def update_theta_f(self, i):
        mu_theta = self.y_tilda.t() / self.sigma2_f 
        sigma2_theta = 1 / (1 / self.sigma2_f + 1 / self.lamb)
        sigma2_theta = sigma2_theta.repeat(self.n, 1)

        dist = torch.distributions.Normal(mu_theta, torch.sqrt(sigma2_theta))
        self.theta_f_batch = dist.sample()
        self.theta_f_batch = self.theta_f_batch.t()
    

    def update_theta_rho(self, i):
        omega = self.Psi_detach @ self.theta_rho
        #omega = torch.from_numpy(random_polyagamma(z = omega.detach().numpy()).astype(np.float32))
        omega = torch.from_numpy(pgdraw_f(1, ro.FloatVector(omega.detach().numpy()))).float()
        cov = self.Psi_detach.t() @ torch.diag(omega) @ self.Psi_detach
        cov += torch.diag(1/self.lamb)
        # cov = torch.inverse(cov)
        # mu = cov @ self.Psi_detach.t() @ (self.delta - 0.5) 

        # dist = torch.distributions.MultivariateNormal(mu, cov)
        # self.theta_rho = dist.sample()
        # self.logp_theta_rho[i] = dist.log_prob(self.theta_rho)
        precision = cov
        mu = self.Psi_detach.t() @ (self.delta - 0.5) 
        R = torch.linalg.cholesky(precision, upper=True)
        b = torch.linalg.solve(R.t(), mu)
        Z = torch.randn(self.J)
        self.theta_rho = torch.linalg.solve(R, Z+b)

    def update_theta_beta(self, i):
        temp = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) - self.X_batch @ self.alpha
        self.Y_delta_Phi = self.y_batch[self.delta==1].t() @ self.Psi_detach[self.delta==1]  ## N by L
        self.sigma2_theta_beta = 1 / (torch.sum(self.Y_delta_Phi ** 2, 0) / self.sigma2_u +  1/self.lamb)
        for l in range(self.J):
            temp += self.Y_delta_Phi[:,l] * self.theta_beta[l]
            self.mu_alpha = self.sigma2_theta_beta[l] * torch.sum(temp * self.Y_delta_Phi[:,l]) / self.sigma2_u
            dist = torch.distributions.Normal(self.mu_alpha, torch.sqrt(self.sigma2_theta_beta[l]))
            self.theta_beta[l] = dist.sample()
            temp -= self.Y_delta_Phi[:,l] * self.theta_beta[l]
        self.update_beta() 

    

    def update_delta(self):
        u = self.Psi_detach @ self.theta_rho
        p1 = log1pexp(-u)
        p0 = log1pexp(u)
        u_res = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) - self.X_batch @ self.alpha
        for v in range(self.V):
            u_res += self.y_batch[v:v+1,].t()@ (self.beta[v:v+1] * self.delta[v])
                
            temp0 = u_res
            temp1 = u_res - self.y_batch[v:v+1,].t()@ self.beta[v:v+1]

            temp0 = -0.5 * torch.sum(temp0 ** 2) / self.sigma2_u - p0[v]
            temp1 = -0.5 * torch.sum(temp1 ** 2) / self.sigma2_u - p1[v]
            logp = torch.column_stack((temp0, temp1))
            logp_max, ind = torch.max(logp, 1)
            prob = torch.exp(logp - logp_max.unsqueeze_(1))
            pp = prob[:,1] / torch.sum(prob, 1)
            self.delta[v] = torch.bernoulli(pp)

            u_res -= self.y_batch[v:v+1,].t()@ (self.beta[v:v+1] * self.delta[v])  


    def update_alpha(self, i):
        temp = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) -  self.X_batch @ self.alpha
        sigma2_alpha = 1 / (self.ssq_X_batch/self.sigma2_u + 1/self.sigma2_alpha)
        for j in range(self.q):
            temp += self.X_batch[:,j] * self.alpha[j]
            mu_beta = sigma2_alpha[j] * torch.sum(temp * self.X_batch[:,j]) / self.sigma2_u
            #print(mu_beta)
            dist = torch.distributions.Normal(mu_beta, torch.sqrt(sigma2_alpha[j]))
            self.alpha[j] = dist.sample()
            temp -= self.X_batch[:,j] * self.alpha[j]
 
    
    def update_beta(self):
        self.beta = self.Psi_detach @ self.theta_beta
    

    def update_theta_nn(self, i, k):
        log_prior = - 0.5 * torch.sum(self.theta_nn ** 2)/ self.sigma2_nn
        log_ll = -0.5 * torch.sum((self.U_batch - self.y_batch.t() @ ((self.Psi @ self.theta_beta) * self.delta) -  self.X_batch @ self.alpha) ** 2)/ self.sigma2_u
        log_post = log_prior + self.N / self.n * log_ll
        grad_log = torch.autograd.grad(log_post/self.N, self.theta_nn)[0]
    
        with torch.no_grad():
            self.theta_nn +=  0.5 * self.lrt * grad_log + self.eta[k]
            self.theta_nn.grad = None
        self.return_weights()
        self.nn_out = self.model(self.S)
        if self.ortho == 'GS':
            self.update_Psi_GS(i)
        elif self.ortho == 'SVD':
            self.update_Psi_SVD()
        self.update_beta()
        self.update_y_tilda()
    
    
    def update_sigma2_f(self):
        a_f_new = self.V * self.n / 2 + self.a_f
        b_f_new = self.rss_y / 2 + self.b_f
        m = torch.distributions.Gamma(a_f_new, b_f_new)
        self.sigma2_f = 1 / m.sample()

    def update_sigma2_u(self):
        a_u_new = self.n / 2 + self.a_u
        b_u_new = self.rss_u / 2 + self.b_u
        m = torch.distributions.Gamma(a_u_new, b_u_new)
        self.sigma2_u = 1 / m.sample()
    
    def update_sigma2_nn(self):
        a_nn_new = self.theta_nn_cumsum_ind[-1] / 2 + self.a_nn
        b_nn_new = torch.sum(self.theta_nn ** 2) / 2 + self.b_nn
        m = torch.distributions.Gamma(a_nn_new, b_nn_new)
        self.sigma2_nn = 1 / m.sample()

    def update_sigma2_alpha(self):
        a_alpha_new = self.q / 2 + self.a_alpha
        b_alpha_new = torch.sum(self.alpha ** 2) / 2 + self.b_alpha
        m = torch.distributions.Gamma(a_alpha_new, b_alpha_new)
        self.sigma2_alpha = 1 / m.sample()

    

    def update_loglik_u(self,i):
        u_mean = self.y.t() @ (self.beta * self.delta)  + self.X @ self.alpha
        u_sd = torch.sqrt(self.sigma2_u)
        u_dist = torch.distributions.Normal(u_mean, u_sd)
        self.log_lik_u[i] = torch.sum(u_dist.log_prob(self.U))
    
    def update_loglik_y(self,i):
        y_mean = self.f_est.detach()
        y_sd = torch.sqrt(self.sigma2_f)
        y_dist = torch.distributions.Normal(y_mean, y_sd)
        self.log_lik_y[i] = torch.sum(y_dist.log_prob(self.y))

 

    def set_mcmc_samples(self):
        self.mcmc_f = torch.zeros(self.mcmc_sample, self.V, self.N)
        self.mcmc_theta_nn = torch.zeros(self.theta_nn_cumsum_ind[-1])
        self.mcmc_theta_f = torch.zeros(self.mcmc_sample, self.J, self.N)
        self.mcmc_theta_beta = torch.zeros(self.mcmc_sample, self.J)
        self.mcmc_alpha = torch.zeros(self.mcmc_sample, self.q)
        
        self.mcmc_beta = torch.zeros(self.mcmc_sample,self.V)
        self.mcmc_delta = torch.zeros(self.mcmc_sample,self.V)

        self.mcmc_sigma2_u = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_f = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_nn = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_alpha = torch.zeros(self.mcmc_sample)

       
        self.mcmc_theta_rho = torch.zeros(self.mcmc_sample, self.J)
        

    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_f[mcmc_iter,:,:] = self.f_est.detach()
        self.mcmc_theta_nn += self.theta_nn.detach()
        self.mcmc_theta_f[mcmc_iter,:,:] = self.theta_f.detach()
        self.mcmc_theta_beta[mcmc_iter,:] = self.theta_beta.detach()
        self.mcmc_alpha[mcmc_iter,:] = self.alpha
        
        self.mcmc_delta[mcmc_iter,:] = self.delta
        self.mcmc_beta[mcmc_iter,:] = self.beta
       
        self.mcmc_sigma2_u[mcmc_iter] = self.sigma2_u
        self.mcmc_sigma2_f[mcmc_iter] = self.sigma2_f
        self.mcmc_sigma2_alpha[mcmc_iter] = self.sigma2_alpha
        self.mcmc_sigma2_nn [mcmc_iter]= self.sigma2_nn
        self.mcmc_theta_rho[mcmc_iter,:] = self.theta_rho
       
    def post_summary(self, level=0.05):
        post_alpha = torch.median(self.mcmc_alpha, 0)[0]
        post_delta = torch.mean(self.mcmc_delta, 0)
        post_beta = torch.median(self.mcmc_beta, 0)[0]
       
        post_sigma2_nn = torch.median(self.mcmc_sigma2_nn, 0)[0]
        post_sigma2_alpha = torch.median(self.mcmc_sigma2_alpha, 0)[0]
        post_sigma2_f = torch.median(self.mcmc_sigma2_f, 0)[0]
        post_sigma2_u = torch.median(self.mcmc_sigma2_u, 0)[0]

        prob_fdr, _, _ = FDR_SI(post_delta, level)
        maineff = (post_beta * prob_fdr).t()

        post_params = {'maineff': maineff,
                       "post_beta": post_beta,
                       "post_delta": post_delta,
                       "post_alpha": post_alpha,
                       "post_sigma2_u": post_sigma2_u,
                       "post_sigma2_f": post_sigma2_f,
                       "post_sigma2_alpha": post_sigma2_alpha,
                       "post_sigma2_nn": post_sigma2_nn,
                    }

        return post_params


    def set_loglik(self):
        self.log_lik_y = torch.zeros(self.total_iter)
        self.log_lik_u = torch.zeros(self.total_iter)


    
    def predict(self, Y_test=None, X_test=None):
        if Y_test is None:
            Y_test = self.y_test
        if X_test is None:
            X_test = self.X_test
        
        ndraw = self.mcmc_sample
        test_n = Y_test.shape[1]
        pred_U = torch.zeros(ndraw, test_n)
        for i in range(ndraw):
            u_pred_mean = Y_test.t() @ (self.mcmc_beta[i,:]* self.mcmc_delta[i,:]) + X_test @ self.mcmc_alpha[i,:]
            dist=torch.distributions.Normal(u_pred_mean, torch.sqrt(self.mcmc_sigma2_u[i]))
            pred_U[i] = dist.sample()
        return(pred_U)


def get_weights(m):
    param = torch.empty(0)
    for p in m.parameters():
      p_vec = p.reshape(-1)
      param = torch.cat((param, p_vec))
    
    return param

def return_weights(m, param, cumsum_vars):
    for i, p in enumerate(m.parameters()):
        value = (param[cumsum_vars[i]:cumsum_vars[i+1]]).clone().detach().requires_grad_(True).reshape(p.shape)
        p.data = value
    return(m)