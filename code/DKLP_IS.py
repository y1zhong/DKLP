import torch
import numpy as np
from torch import nn
import math
from helper import log1pexp, DKLP_NN
from tqdm import tqdm
from polyagamma import random_polyagamma

class IS():
    def __init__(self, Y, X, S,
                L=50, H=128, M=4, act_fn='relu',
                init_theta_nn = None, 
                init_theta_beta = None, 
                init_theta_eta = None,
                init_theta_rho = None,
                init_delta = None,
                init_sigma2_eps = None,
                init_sigma2_nn = None, 
                init_lamb = None,
                init_a_lamb = None,
                intercept = 1,
                diminishing_ratio= 0.1, r = 0.55,
                a_eps=0.01, b_eps=0.01,
                a_nn=0.01, b_nn=0.01, 
                A2=100,
                first_burnin=100, second_burnin = 100, 
                thin=1, mcmc_sample=100, 
                lr=0.01, batch_size = None
                ):
        self.y = Y # image output, V by N
        self.X = X # num of covariates, N by J
        self.Xt = X.t()
        self.S = S # image coordinates, V by d
        
       
        self.V = S.shape[0] # num of locations
        self.d = S.shape[1] # grid dimension

       
        self.N = Y.shape[1] # num of images/lines
        self.L = L
        self.J = X.shape[1]
        self.p = self.J
        self.intercept = intercept
        if(intercept == 1):
            self.p = self.J - 1
        

        ### mcmc settings
        self.first_burnin = first_burnin ### burnin without sparsity
        self.second_burnin = second_burnin ### burnin with sparsity
        self.mcmc_burnin = first_burnin + second_burnin 
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample 
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning

        ### indices split for mini-batch update
        if batch_size is None:
            batch_size = self.N
        self.n = min(batch_size, self.N)
        self.ind_split = torch.split(torch.arange(0, self.N), self.n)
        self.batch_num = len(self.ind_split)

        ### precalcualtion
        self.ssq_X_list = [torch.sum(X[self.ind_split[k]] ** 2, 0) for k in range(self.batch_num)]
        self.X_rescale_list = [- X[self.ind_split[k],:] @ torch.inverse(self.Xt[:, self.ind_split[k]] @ X[self.ind_split[k],:]) @ self.Xt[:, self.ind_split[k]] for k in range(self.batch_num)]
        
        #hyper parameters
        self.lr = lr #learning reate 
        self.b_0 = self.total_iter/((1.0/diminishing_ratio)**(1.0/r) - 1.0)
        self.a_0 = lr*self.b_0**(r)
        self.r = r

        self.V_t=0
        self.m_t=0


        self.a_eps = a_eps
        self.b_eps = b_eps
        self.a_nn = a_nn
        self.b_nn = b_nn
        self.A2 = A2
        
  
        self.model = DKLP_NN(self.d, H, L, M=M, act_fn=act_fn)
        self.theta_nn_cumsum_ind = np.cumsum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.theta_nn_cumsum_ind = np.concatenate(([0], self.theta_nn_cumsum_ind))
        
       
        #initialization
        if init_theta_nn is None:
            init_theta_nn = torch.randn(self.theta_nn_cumsum_ind[-1])

        if init_theta_beta is None:
            init_theta_beta = torch.randn(L, self.J)
    
        if init_theta_eta is None:
            init_theta_eta =  torch.randn(L, self.N)

        if init_sigma2_eps is None:
            init_sigma2_eps =  torch.tensor(1)

        if init_sigma2_nn is None:
            init_sigma2_nn = torch.tensor(1)

        if init_delta is None:
            init_delta = torch.ones(self.V, self.J)
        
        if init_theta_rho is None:
            init_theta_rho = torch.randn(L, self.p)

        if init_lamb is None:
            init_lamb = torch.ones(self.L)
        
        if init_a_lamb is None:
            init_a_lamb = torch.ones(self.L)

        self.theta_nn =  init_theta_nn.requires_grad_()
        self.theta_beta = init_theta_beta
        self.theta_nn =  init_theta_nn
        self.theta_eta = init_theta_eta
        self.sigma2_eps = init_sigma2_eps
        self.sigma2_nn = init_sigma2_nn
        self.delta = init_delta
        self.lamb = init_lamb
        self.a_lamb = init_a_lamb
        self.theta_rho = init_theta_rho


        self.return_weights()
        self.nn_out = self.model(self.S)
        self.update_Psi()
        self.beta = self.Psi_detach  @ self.theta_beta
        self.eta = self.Psi_detach @ self.theta_eta

        # intermediate parameters initialization
        self.f_est = torch.zeros(self.V, self.N) # est functions, VxK
        self.rss = torch.zeros(1) # residual sum of suqares
        

        self.set_mcmc_samples()
        self.set_loglik()


    def fit(self):
        for i in tqdm(range(self.total_iter)):
            self.lrt = self.a_0*(self.b_0 + i)**(-self.r) 
            self.epsilon = torch.randn(self.batch_num) * math.sqrt(self.lrt)
            for k in range(self.batch_num):
                self.batch_ind = self.ind_split[k]
                self.update_batch(k)

                self.update_theta_nn(i, k)
                self.update_lamb(i)
                self.update_a_lamb(i)
                
                self.update_theta_eta(i)
                self.theta_eta[:, self.batch_ind] = self.theta_eta_batch
                self.eta[:, self.batch_ind] = self.eta_batch
                
                self.update_theta_beta_select(i)

                if((i >= self.first_burnin) & (self.second_burnin != 0)):
                        self.update_theta_rho(i)
                        self.update_delta_select()

                self.update_f()
                self.f_est[:, self.batch_ind] = self.f_est_batch.detach()
                
                self.update_sigma2_eps()
                self.update_sigma2_nn()
                
            self.update_loglik(i)
            if i >= self.mcmc_burnin:
                if (i - self.mcmc_burnin) % (self.mcmc_thinning) == 0:
                    mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                    self.save_mcmc_samples(mcmc_iter)
                

    def update_f(self):
        self.f_est_batch = (self.beta * self.delta.detach())  @ self.Xt_batch + self.Phi_detach @ self.theta_eta_batch
        self.rss = torch.sum((self.y_batch - self.f_est_batch) ** 2)

    def update_Psi(self):
        self.Psi, R = torch.linalg.qr(self.nn_out)
        R_diag = R.diag()
        for l in range(self.L) :
            if(R_diag[l] < 0):
                self.Psi[:,l] = self.Psi[:,l] * (-1)
                R_diag[l] = R_diag[l] * (-1)
        self.Psi_detach = self.Psi.detach()
        self.Phi = self.Psi
        self.Phi_detach = self.Phi.detach()

    def update_lamb(self,i):
        a_eps_new = (1 + self.J + self.n ) / 2 
        b_eps_new = (torch.sum(self.theta_beta ** 2, 1) +  torch.sum(self.theta_eta ** 2, 1))/ 2 + 1 / self.a_lamb
        if i >= self.first_burnin:
            a_eps_new += self.p / 2 
            b_eps_new += torch.sum(self.theta_rho ** 2, 1) / 2
            
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.lamb = 1 / m.sample()

    def update_a_lamb(self,i):
        b_eps_new = 1 / self.A2 + 1 / self.lamb
        m = torch.distributions.Gamma(1, b_eps_new)
        self.a_lamb = 1 / m.sample()


    def update_beta(self):
        self.beta = self.Psi_detach @ self.theta_beta

    def update_eta(self):
        self.eta_batch = self.Psi_detach @ self.theta_eta_batch



    def update_delta_select(self):
        u = self.Psi_detach @ self.theta_rho
        if (self.intercept == 1):
            u = torch.column_stack((torch.ones(self.V), u))
        p1 = log1pexp(-u)
        p0 = log1pexp(u)
       
        y_res = self.y_batch - (self.beta * self.delta) @ self.Xt_batch - self.Phi_detach @ self.theta_eta_batch
        for j in range(self.J):
            if (self.intercept == 1) and (j == 0):
                continue
            select_ind = self.delta[:,j]==1
            Psi_sub = self.Psi_detach[select_ind] 
            y_res[select_ind, ] += Psi_sub @ self.theta_beta[:,j:(j+1)] @ self.Xt_batch[j:(j+1),:]
            temp0 = y_res
            temp1 = temp0 - self.Psi_detach @ self.theta_beta[:,j:(j+1)] @ self.Xt_batch[j:(j+1),:]

            temp0 = -0.5 * torch.sum(temp0 ** 2, 1) / self.sigma2_eps - p0[:,j]
            temp1 = -0.5 * torch.sum(temp1 ** 2, 1) / self.sigma2_eps - p1[:,j]
            logp = torch.column_stack((temp0, temp1))
            logp_max, ind = torch.max(logp, 1)
            prob = torch.exp(logp - logp_max.unsqueeze_(1))
            #print(prob)
            pp = prob[:,1] / torch.sum(prob, 1)
            self.delta[:,j] = torch.bernoulli(pp)
            select_ind = self.delta[:,j]==1
            Psi_sub = self.Psi_detach[select_ind] 
            y_res[select_ind, ] -= Psi_sub @ self.theta_beta[:,j:(j+1)] @ self.Xt_batch[j:(j+1),:]


    def update_theta_eta(self,i):
        y_eta = self.y_batch - (self.beta * self.delta.detach()) @ self.Xt_batch
        y_eta = self.Psi_detach.t() @ y_eta

        sigma_eta2 =  1 / (1 / self.sigma2_eps + 1 / self.lamb)
        mu_eta = torch.t(y_eta) / self.sigma2_eps
        
        dist = torch.distributions.Normal(mu_eta, torch.sqrt(sigma_eta2))
        self.theta_eta_batch = dist.sample()
        #self.logp_theta_eta[i] = torch.sum(dist.log_prob(self.theta_eta_batch)).detach()
        self.logp_theta_eta[i] = torch.sum(torch.distributions.Normal(0, 1).log_prob(self.theta_eta_batch))
        self.theta_eta_batch = torch.t(self.theta_eta_batch)
        self.theta_eta_batch += self.theta_eta_batch @ self.X_batch_rescale
        
        self.update_eta()

    def update_theta_rho(self, i):
        self.logp_theta_rho[i] = 0
        omega = self.Psi_detach @ self.theta_rho
        # omega = pgdraw_f(1, rpy2.robjects.r.matrix(omega.numpy(), nrow=self.V, ncol=self.p))
        # omega = omega.reshape(self.p,self.V).transpose()
        # omega = torch.from_numpy(omega).float()
        omega = torch.from_numpy(random_polyagamma(z = omega.detach().numpy()).astype(np.float32))
        for j in range(self.p):
            cov = self.Psi_detach.t() @ torch.diag(omega[:,j]) @ self.Psi_detach
            cov += torch.diag(1/self.lamb)
            precision = cov
            mu = self.Psi_detach.t() @ (self.delta[:,(j + self.intercept)] - 0.5) 
            R = torch.linalg.cholesky(precision, upper=True)
            b = torch.linalg.solve(R.t(), mu)
            Z = torch.randn(self.L)
            self.theta_rho[:,j] = torch.linalg.solve(R, Z+b)

            # cov = torch.inverse(cov)
            # mu = cov @ self.Phi_detach.t() @ (self.delta[:,(j + self.intercept)] - 0.5) 
            # # print(mu)
            # # print(cov)
            # dist = torch.distributions.MultivariateNormal(mu, cov)
            # self.theta_rho[:,j] = dist.sample()


    def update_theta_nn(self, i, k):
        # log_prior_nn = - 0.5 * torch.sum(self.theta_nn ** 2) / self.sigma2_nn  
        # y_res = ((self.Psi @ self.theta_beta) * self.delta.detach()) @ self.Xt_batch + self.Psi @ self.theta_eta_batch
        # log_ll =  - 0.5 * torch.sum((self.y_batch - y_res) ** 2) / self.sigma2_eps

        # log_post = log_prior_nn + self.N / self.n * log_ll
        # grad_log = torch.autograd.grad(log_post/self.N, self.theta_nn)[0]
        log_prior_nn = - 0.5 * torch.sum(self.theta_nn ** 2) / self.sigma2_nn  
        y_res = ((self.Psi @ self.theta_beta) * self.delta.detach()) @ self.Xt_batch + self.Psi @ self.theta_eta_batch
        log_ll =  - 0.5 * torch.sum((self.y_batch - y_res) ** 2) / self.sigma2_eps

        log_post = -(log_prior_nn + self.N / self.n * log_ll)
        du_t = torch.autograd.grad(log_post/self.N, self.theta_nn)[0]


        with torch.no_grad():
            self.theta_nn +=  0.5 * self.lrt * du_t + self.epsilon[k]
            self.theta_nn.grad = None

        self.return_weights()
        self.nn_out = self.model(self.S)
        self.update_Psi()
        self.update_beta()
        self.update_eta()
        self.logp_theta_nn[i] = log_post.detach()  
    
    
    def update_theta_beta_select(self, i):
        y_res = self.y_batch - self.eta_batch - (self.beta * self.delta.detach()) @ self.Xt_batch
        for j in range(self.J):
            # select_ind = (self.delta[:,j]==1).nonzero().flatten().detach()
            # Psi_sub = self.Psi_detach[select_ind] # select by row, on V
            # for l in range(self.L):
            #     y_res[select_ind, ] += (Psi_sub[:,l:(l+1)] * self.theta_beta[l,j]) @ self.Xt_batch[j:(j+1),:]
            #     sigma2_theta_beta_jl = 1 / (torch.sum((Psi_sub[:,l] ** 2) *  self.ssq_X_batch[j])/ self.sigma2_eps +  1/self.lamb[l])
            #     mu_theta_beta_jl = sigma2_theta_beta_jl * Psi_sub[:,l].t() @ y_res[select_ind, ] @ self.X_batch[:,j] / self.sigma2_eps
            #     dist = torch.distributions.Normal(mu_theta_beta_jl, torch.sqrt(sigma2_theta_beta_jl))
            #     self.theta_beta[l,j] = dist.sample()
            #     y_res[select_ind, ] -= (Psi_sub[:,l:(l+1)] * self.theta_beta[l,j]) @ self.Xt_batch[j:(j+1),:]
            #precision = Phi_sub.t() @ Phi_sub * self.ssq_X_batch[j] / self.sigma2_eps + torch.eye(self.L)
            select_ind = (self.delta[:,j]==1).nonzero().flatten().detach()
            Psi_sub = self.Psi_detach[select_ind] # select by row, on V

            y_res[select_ind, ] += Psi_sub @ self.theta_beta[:,j:(j+1)] @ self.Xt_batch[j:(j+1),:]
            #precision = Psi_sub.t() @ Psi_sub * self.ssq_X_batch[j] / self.sigma2_eps + torch.eye(self.L)
            precision = Psi_sub.t() @ Psi_sub * self.ssq_X_batch[j] / self.sigma2_eps + torch.diag(1/self.lamb)

            mu = Psi_sub.t() @ y_res[select_ind, ] @ self.X_batch[:,j] / self.sigma2_eps
            R = torch.linalg.cholesky(precision, upper=True)
            b = torch.linalg.solve(R.t(), mu)
            Z = torch.randn(self.L)
            self.theta_beta[:,j] = torch.linalg.solve(R, Z+b)
            y_res[select_ind, ] -= Psi_sub @ self.theta_beta[:,j:(j+1)] @ self.Xt_batch[j:(j+1),:]
        self.update_beta()
   


    def update_sigma2_eps(self):
        a_eps_new = self.V * self.n / 2 + self.a_eps
        b_eps_new = self.rss.detach() / 2 + self.b_eps
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.sigma2_eps = 1 / m.sample()
    
    def update_sigma2_nn(self):
        a_nn_new = self.theta_nn_cumsum_ind[-1] / 2 + self.a_nn
        b_nn_new = torch.sum(self.theta_nn.detach() ** 2) / 2 + self.b_nn
        m = torch.distributions.Gamma(a_nn_new, b_nn_new)
        self.sigma2_nn = 1 / m.sample()




    def update_loglik(self,i):
        y_dist = torch.distributions.Normal(self.f_est, torch.sqrt(self.sigma2_eps))
        self.log_lik[i] = torch.sum(y_dist.log_prob(self.y))

    def set_mcmc_samples(self):
        self.mcmc_delta = torch.zeros(self.mcmc_sample, self.V, self.J)
        self.mcmc_beta = torch.zeros(self.mcmc_sample, self.V, self.J)
        self.mcmc_eta = torch.zeros(self.mcmc_sample, self.V, self.N)
    
        self.mcmc_lamb = torch.zeros(self.mcmc_sample, self.L)
        self.mcmc_sigma2_eps = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_nn = torch.zeros(self.mcmc_sample)
        #self.mcmc_Psi = torch.zeros(self.mcmc_sample, self.V, self.L)
        # self.mcmc_f = torch.zeros(self.mcmc_sample, self.V, self.N)
        # self.mcmc_Phi = torch.zeros(self.mcmc_sample, self.V, self.L)
        # self.mcmc_theta_nn = torch.zeros(self.theta_nn_cumsum_ind[-1])
        # self.mcmc_theta_beta = torch.zeros(self.mcmc_sample, self.L, self.J)
        # self.mcmc_theta_eta = torch.zeros(self.mcmc_sample, self.L, self.N)

        # if self.threshold == 0: #selection
        #     self.mcmc_theta_rho = torch.zeros(self.mcmc_sample, self.L, self.p)
        # else: #threshold
        #     self.mcmc_beta_prime = torch.zeros(self.mcmc_sample, self.V, self.J)
        #     self.mcmc_zeta = torch.zeros(self.mcmc_sample, self.J)

    

    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_eta[mcmc_iter,:,:]= self.eta
        self.mcmc_delta[mcmc_iter,:,:] = self.delta.detach()
        self.mcmc_beta[mcmc_iter,:,:] = self.beta
        
        self.mcmc_sigma2_eps[mcmc_iter] = self.sigma2_eps
        self.mcmc_sigma2_nn [mcmc_iter]= self.sigma2_nn
        self.mcmc_lamb[mcmc_iter,:] = self.lamb
        #self.mcmc_Psi[mcmc_iter,:,:] = self.Psi_detach
        # self.mcmc_f[mcmc_iter,:,:] = self.f_est.detach()
        # self.mcmc_Phi[mcmc_iter,:,:] = self.Phi.detach()
        # self.mcmc_theta_nn += self.theta_nn.detach()
        # self.mcmc_theta_beta[mcmc_iter,:,:] = self.theta_beta.detach()
        # self.mcmc_theta_eta[mcmc_iter,:,:] = self.theta_eta
        
        

        # if self.threshold == 0: #selection
        #     self.mcmc_theta_rho[mcmc_iter,:,:] = self.theta_rho.detach()
        # else: #threshold
        #     self.mcmc_zeta[mcmc_iter,:] = self.zeta
        #     self.mcmc_beta_prime[mcmc_iter,:,:] = self.beta_prime

    def set_loglik(self):
        self.logp_theta_nn = torch.zeros(self.total_iter)
        self.logp_theta_beta = torch.zeros(self.total_iter)
        self.logp_theta_eta = torch.zeros(self.total_iter)
        self.log_lik = torch.zeros(self.total_iter)
        self.logp_theta_rho = torch.zeros(self.total_iter)

    def update_batch(self, k):
        self.y_batch = self.y[:, self.batch_ind]
        self.X_batch = self.X[self.batch_ind]
        self.Xt_batch = self.Xt[:, self.batch_ind]

        self.ssq_X_batch = self.ssq_X_list[k]
        self.eta_batch = self.eta[:, self.batch_ind]
        self.theta_eta_batch = self.theta_eta[:, self.batch_ind]
        self.f_est_batch = self.f_est[:, self.batch_ind]
        self.X_batch_rescale = self.X_rescale_list[k]

    def post_mean_est(self):
        post_delta = torch.mean(self.mcmc_delta, 0)
        post_beta = torch.mean(self.mcmc_beta, 0)
        post_eta = torch.mean(self.mcmc_eta, 0)
       
        post_sigma2_nn = torch.mean(self.mcmc_sigma2_nn)
        post_sigma2_eps = torch.mean(self.mcmc_sigma2_eps)
        post_lamb = torch.mean(self.mcmc_lamb, 0)
        #post_Psi = torch.mean(self.mcmc_Psi, 0)
        # return post_thresh_alpha, post_theta_f, post_theta_alpha, post_zeta, post_beta, post_theta_nn, post_f, post_u, post_sigma2_u, post_sigma2_f, post_sigma2_beta, post_sigma2_nn 
        return post_beta, post_delta, post_eta, post_sigma2_nn, post_sigma2_eps, post_lamb


    def post_CI(self):
        f_lci = torch.quantile(self.mcmc_f, 0.025, 0)
        f_uci = torch.quantile(self.mcmc_f, 0.975, 0)
        return f_lci, f_uci

    def get_weights(self):
        param = torch.empty(0)
        for p in self.model.parameters():
            p_vec = p.reshape(-1)
            param = torch.cat((param, p_vec))
        return param
    
    def return_weights(self):
        for i, p in enumerate(self.model.parameters()):
            value = (self.theta_nn[self.theta_nn_cumsum_ind[i]:self.theta_nn_cumsum_ind[i+1]]).clone().detach().reshape(p.shape).requires_grad_()
            p.data = value
    




# class DKLP_NN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.body = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim)
#         )
#     def forward(self, x):
#         y = self.body(x)
#         return y
    


