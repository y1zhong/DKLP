#### scalar on image regression, SGLD, thresholding and selection
import torch
import numpy as np
from torch import nn
#import matplotlib.pyplot as plt
import math
# from polyagamma import random_polyagamma
from tqdm import tqdm
from scipy.stats import truncnorm
from helper import log1pexp, DKLP_NN
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, data
import rpy2.robjects.numpy2ri
ro.numpy2ri.activate()
pgdraw = importr('pgdraw')
pgdraw_f = rpy2.robjects.r['pgdraw']



class SI():
    def __init__(self, grid, Y, U, X,
                 L=50, H=128, M=4, act_fn='relu',
                init_theta_nn=None, init_theta_f=None,
                init_theta_beta=None, init_alpha=None,
                init_sigma2_nn=None, init_sigma2_u=None,
                init_sigma2_f=None, init_sigma2_alpha=None,
                init_theta_rho=None, init_delta = None,
                init_lamb=None,init_a_lamb=None,
                include_intercept=0,
                batch_size = 100, 
                a_f = 0.01, b_f = 0.01,
                a_u = 0.01, b_u = 0.01,
                a_nn = 0.01, b_nn = 0.01,
                a_alpha = 0.01, b_alpha = 0.01, A2=100,
                kernel_burnin = 200, selection_burnin=200, thin = 1, mcmc_sample = 500,
                lr = 0.001, diminishing_ratio= 0.1, r = 0.55
                ):
        
        #### need some dimension check ####

        self.y = Y # image output, V by N
        self.X = X # other covariates, N by J
        self.S = grid
        self.V = grid.shape[0] # num of locations
        self.d = grid.shape[1] # grid dimension
        self.N = Y.shape[1] # num of images/lines
        self.L = L
        self.U = U # scalar response variable, N by 1
     

        self.Xt = torch.t(self.X) # num of covariates, J by N
        self.ssq_X = torch.sum(self.X ** 2, 0)
        self.J = self.X.shape[1]

        #self.ns = math.sqrt(self.V) ### normalizing scalar
        self.ns=1
        
        
        ### indices split for batch update
        self.n = min(batch_size, self.N)
        self.ind_split = torch.split(torch.arange(0, self.N), self.n)
        self.num_split = len(self.ind_split)

        ### mcmc settings
        self.kernel_burnin = kernel_burnin
        self.selection_burnin = selection_burnin
        self.mcmc_burnin = kernel_burnin + selection_burnin 
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample 
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning
        
        #hyper parameters
        self.lr = lr #learning reate
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

        
        #initialization
        self.model = DKLP_NN(self.d, H, L, M=M, act_fn=act_fn)
        self.theta_nn_cumsum_ind = np.cumsum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.theta_nn_cumsum_ind = np.concatenate(([0], self.theta_nn_cumsum_ind))

        if init_theta_nn is None:
            init_theta_nn = torch.randn(self.theta_nn_cumsum_ind[-1])

        if init_theta_beta is None:
            init_theta_beta = torch.randn(L)
    
        if init_theta_f is None:
            init_theta_f =  torch.randn(L, self.N)

        if init_alpha is None:
            init_alpha = torch.randn(self.J)

        
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
           init_theta_rho = torch.randn(self.L)
        
        if init_lamb is None:
           init_lamb = torch.ones(self.L)

        if init_a_lamb is None:
           init_a_lamb = torch.ones(self.L)

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
        self.update_Psi()
        self.beta = self.Psi_detach @ self.theta_beta
        
        self.y_tilda = torch.zeros(self.L, self.N) # y transform, LxN
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

                    #self.update_Phi()
                    self.update_theta_nn(i,k) #failed
                    
                    #self.update_theta_f(i) 
                    #self.theta_f[:, self.batch_ind] = self.theta_f_batch
                    self.update_theta_beta_selection(i) # checked
                    self.update_lamb(i)
                    self.update_a_lamb()
                    self.update_alpha(i) # checked
                    
                    if(i >= self.kernel_burnin):
                        self.update_theta_rho(i)
                        self.update_delta_select()
                    
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
                    # self.save_mcmc_trace(i)
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
    
    def update_Psi(self):
        # self.Psi, lambda_sqrt, Vt = torch.linalg.svd(self.nn_out, full_matrices=False)
        # #print(self.U)
        # for l in range(self.L) :
        #     if(self.Psi[0,l] < 0):
        #         self.Psi[:,l] = self.Psi[:,l] *(-1)

        # self.Phi = self.Psi
        # self.Phi_detach = self.Phi.detach()
        # self.Psi_detach = self.Psi.detach()

        Q, R = torch.linalg.qr(self.nn_out)
        R_diag = R.diag()
        for l in range(self.L) :
            if(R_diag[l] < 0):
                Q[:,l] = Q[:,l] * (-1)
                R_diag[l] = R_diag[l] * (-1)
        self.Psi = Q
        self.Psi_detach = self.Psi.detach()
        self.Phi = self.Psi
        self.Phi_detach = self.Psi_detach
        #self.update_Phi()

    # def update_Phi(self):
    #     self.Phi = (self.Psi @ torch.diag(torch.sqrt(self.lamb)))
    #     self.Phi_detach = self.Phi.detach()

    def update_lamb(self,i):

        #a_eps_new = (1 + 1 + self.n ) / 2 
        a_eps_new = (1 + 1  ) / 2 
        #b_eps_new = ( torch.sum(self.theta_f ** 2, 1))/ 2 + 1 / self.a_lamb
        b_eps_new = (self.theta_beta ** 2) / 2 + 1 / self.a_lamb
        if i >= self.kernel_burnin:
            a_eps_new += 1 / 2 
            b_eps_new += (self.theta_rho ** 2) / 2
        m = torch.distributions.Gamma(a_eps_new, b_eps_new)
        self.lamb = 1 / m.sample()
        #self.update_Phi()

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
        self.logp_theta_f[i] = torch.sum(dist.log_prob(self.theta_f_batch.detach()))
        self.theta_f_batch = self.theta_f_batch.t()
    

    def update_theta_rho(self, i):
        self.logp_theta_rho[i] = 0
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
        Z = torch.randn(self.L)
        self.theta_rho = torch.linalg.solve(R, Z+b)

    def update_theta_beta_selection(self, i):
        temp = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) - self.X_batch @ self.alpha
        self.Y_delta_Phi = self.y_batch[self.delta==1].t() @ self.Psi_detach[self.delta==1]  ## N by L
        self.sigma2_theta_beta = 1 / (torch.sum(self.Y_delta_Phi ** 2, 0) / self.sigma2_u +  1/self.lamb)
        for l in range(self.L):
            temp += self.Y_delta_Phi[:,l] * self.theta_beta[l]
            self.mu_alpha = self.sigma2_theta_beta[l] * torch.sum(temp * self.Y_delta_Phi[:,l]) / self.sigma2_u
            dist = torch.distributions.Normal(self.mu_alpha, torch.sqrt(self.sigma2_theta_beta[l]))
            self.theta_beta[l] = dist.sample()
            self.logp_beta[i] += dist.log_prob(self.theta_beta[l])
            temp -= self.Y_delta_Phi[:,l] * self.theta_beta[l]
        self.update_beta() 

    

    def update_delta_select(self):
        u = self.Psi_detach @ self.theta_rho
        p1 = log1pexp(-u)
        p0 = log1pexp(u)
       
        u_res = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) - self.X_batch @ self.alpha
        for v in range(self.V):
            #Q_v = self.theta_f_batch.t() @ self.Phi[v:v+1,].t()@ self.Phi[v:v+1,] @ self.theta_alpha / self.V
            u_res += self.y_batch[v:v+1,].t()@ (self.beta[v:v+1] * self.delta[v])
                
            temp0 = u_res
            temp1 = u_res - self.y_batch[v:v+1,].t()@ self.beta[v:v+1]

            temp0 = -0.5 * torch.sum(temp0 ** 2) / self.sigma2_u - p0[v]
            temp1 = -0.5 * torch.sum(temp1 ** 2) / self.sigma2_u - p1[v]
            logp = torch.column_stack((temp0, temp1))
            logp_max, ind = torch.max(logp, 1)
            prob = torch.exp(logp - logp_max.unsqueeze_(1))
            #print(prob)
            pp = prob[:,1] / torch.sum(prob, 1)
            self.delta[v] = torch.bernoulli(pp)

            u_res -= self.y_batch[v:v+1,].t()@ (self.beta[v:v+1] * self.delta[v])  


    def update_alpha(self, i):
        self.logp_beta[i] = 0
        temp = self.U_batch - self.y_batch.t() @ (self.beta * self.delta) -  self.X_batch @ self.alpha
        sigma2_alpha = 1 / (self.ssq_X_batch/self.sigma2_u + 1/self.sigma2_alpha)
        for j in range(self.J):
            temp += self.X_batch[:,j] * self.alpha[j]
            mu_beta = sigma2_alpha[j] * torch.sum(temp * self.X_batch[:,j]) / self.sigma2_u
            #print(mu_beta)
            dist = torch.distributions.Normal(mu_beta, torch.sqrt(sigma2_alpha[j]))
            self.alpha[j] = dist.sample()
            self.logp_beta[i] += dist.log_prob(self.alpha[j].detach())
            temp -= self.X_batch[:,j] * self.alpha[j]
 
    
    def update_beta(self):
        self.beta = self.Psi_detach @ self.theta_beta
    

    


    def update_theta_nn(self, i, k):
        log_prior = - 0.5 * torch.sum(self.theta_nn ** 2)/ self.sigma2_nn
        #log_ll = - 0.5 * torch.sum((self.y_batch - self.Psi @ self.theta_f_batch) ** 2) / self.sigma2_f
        log_ll = -0.5 * torch.sum((self.U_batch - self.y_batch.t() @ ((self.Psi @ self.theta_beta) * self.delta) -  self.X_batch @ self.alpha) ** 2)/ self.sigma2_u
        log_post = log_prior + self.N / self.n * log_ll
        grad_log = torch.autograd.grad(log_post/self.N, self.theta_nn)[0]
    
        with torch.no_grad():
            self.theta_nn +=  0.5 * self.lrt * grad_log + self.eta[k]
            self.theta_nn.grad = None
        self.return_weights()
        self.nn_out = self.model(self.S)
        self.update_Psi()
        self.update_beta()
        self.update_y_tilda()
 
        self.logp_theta_nn[i] = log_post.detach()/self.N
    
    
    def update_sigma2_f(self):
        a_f_new = self.V * self.n / 2 + self.a_f
        b_f_new = self.rss_y / 2 + self.b_f
        m = torch.distributions.Gamma(a_f_new, b_f_new)
        self.sigma2_f = 1 / m.sample()

    def update_sigma2_u(self):
        a_u_new = self.n / 2 + self.a_u
        b_u_new = self.rss_u / 2 + self.b_u
        # print(self.rss_u)
        m = torch.distributions.Gamma(a_u_new, b_u_new)
        self.sigma2_u = 1 / m.sample()
    
    def update_sigma2_nn(self):
        a_nn_new = self.theta_nn_cumsum_ind[-1] / 2 + self.a_nn
        b_nn_new = torch.sum(self.theta_nn ** 2) / 2 + self.b_nn
        m = torch.distributions.Gamma(a_nn_new, b_nn_new)
        self.sigma2_nn = 1 / m.sample()

    def update_sigma2_alpha(self):
        a_alpha_new = self.J / 2 + self.a_alpha
        b_alpha_new = torch.sum(self.alpha ** 2) / 2 + self.b_alpha
        # print(torch.sum(self.beta ** 2))
        m = torch.distributions.Gamma(a_alpha_new, b_alpha_new)
        self.sigma2_alpha = 1 / m.sample()


    

    def update_loglik_u(self,i):
        #u_mean = self.y.t() @ self.thresh_alpha / self.V + self.X @ self.beta
        u_mean = self.y.t() @ (self.beta * self.delta)  + self.X @ self.alpha
        u_sd = torch.sqrt(self.sigma2_u)
        u_dist = torch.distributions.Normal(u_mean, u_sd)
        self.log_lik_u[i] = torch.sum(u_dist.log_prob(self.U))
    
    def update_loglik_y(self,i):
        y_mean = self.f_est.detach()
        y_sd = torch.sqrt(self.sigma2_f)
        y_dist = torch.distributions.Normal(y_mean, y_sd)
        self.log_lik_y[i] = torch.sum(y_dist.log_prob(self.y))

    # def pred_f(self):
    #     lam_sqrt = torch.diag(torch.sqrt(self.eig_val))
    #     X_pred = torch.linalg.matmul(self.U[self.test_idx], lam_sqrt)
    #     self.f_pred = torch.linalg.matmul(X_pred, self.theta)

    def set_mcmc_samples(self):
        self.mcmc_f = torch.zeros(self.mcmc_sample, self.V, self.N)
        #self.mcmc_u = torch.zeros(self.mcmc_sample, self.N)
        
        #self.mcmc_lamb = torch.zeros(self.mcmc_sample,self.L)
        #self.mcmc_Phi = torch.zeros(self.mcmc_sample, self.V, self.L)
        self.mcmc_theta_nn = torch.zeros(self.theta_nn_cumsum_ind[-1])
        self.mcmc_theta_f = torch.zeros(self.mcmc_sample, self.L, self.N)
        self.mcmc_theta_beta = torch.zeros(self.mcmc_sample, self.L)
        self.mcmc_alpha = torch.zeros(self.mcmc_sample, self.J)
        
        self.mcmc_beta = torch.zeros(self.mcmc_sample,self.V)
        self.mcmc_delta = torch.zeros(self.mcmc_sample,self.V)

        self.mcmc_sigma2_u = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_f = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_nn = torch.zeros(self.mcmc_sample)
        self.mcmc_sigma2_alpha = torch.zeros(self.mcmc_sample)

       
        self.mcmc_theta_rho = torch.zeros(self.mcmc_sample, self.L)
        

    def save_mcmc_samples(self, mcmc_iter):
        
        self.mcmc_f[mcmc_iter,:,:] = self.f_est.detach()
        # self.mcmc_u[mcmc_iter,:] = self.u_est.detach()

        # self.mcmc_lamb[mcmc_iter,:] = self.lamb.detach()
        # self.mcmc_Phi[mcmc_iter,:,:] = self.Phi.detach()
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
       
    def post_mean_est(self):
        post_alpha = torch.mean(self.mcmc_alpha, 0)
        post_delta = torch.mean(self.mcmc_delta, 0)
        post_beta = torch.mean(self.mcmc_beta, 0)
       
        post_sigma2_nn = torch.mean(self.mcmc_sigma2_nn)
        post_sigma2_alpha = torch.mean(self.mcmc_sigma2_alpha)
        post_sigma2_f = torch.mean(self.mcmc_sigma2_f)
        post_sigma2_u = torch.mean(self.mcmc_sigma2_u )
        # return post_thresh_alpha, post_theta_f, post_theta_alpha, post_zeta, post_beta, post_theta_nn, post_f, post_u, post_sigma2_u, post_sigma2_f, post_sigma2_beta, post_sigma2_nn 
        return post_beta, post_delta, post_alpha, post_sigma2_u, post_sigma2_f, post_sigma2_alpha, post_sigma2_nn 

    def post_mean_outcome(self):
        post_f = torch.mean(self.mcmc_f, 0)
        #post_u = torch.mean(self.mcmc_u, 0)
        #return post_u, post_f
        return  post_f
    
    def post_mean_basis(self):
        post_Phi = torch.mean(self.mcmc_Phi, 0)
        post_lamb = torch.mean(self.mcmc_lamb, 0)
        post_theta_nn = self.mcmc_theta_nn / self.mcmc_sample
        post_theta_f = torch.mean(self.mcmc_theta_f, 0)
        return post_Phi, post_lamb, post_theta_nn, post_theta_f


    def set_loglik(self):
        self.logp_theta_nn = torch.zeros(self.total_iter)
        self.logp_theta_f = torch.zeros(self.total_iter)
        self.logp_beta = torch.zeros(self.total_iter)
        self.logp_alpha = torch.zeros(self.total_iter)
        self.log_lik_y = torch.zeros(self.total_iter)
        self.log_lik_u = torch.zeros(self.total_iter)

        self.logp_theta_rho = torch.zeros(self.total_iter)


    def post_res(self):
        post_alpha = torch.mean(self.mcmc_alpha, 0)
        post_delta = torch.mean(self.mcmc_delta, 0)
        #post_prob = torch.mean(self.mcmc_prob, 0)
        post_alpha_prime = torch.mean(self.mcmc_alpha_prime, 0)
        post_beta = torch.mean(self.mcmc_beta, 0)
        post_zeta = torch.mean(self.mcmc_zeta, )
        post_u = torch.mean(self.mcmc_u, 0)
        post_sigma2_nn = torch.mean(self.mcmc_sigma2_nn)
        post_sigma2_beta = torch.mean(self.mcmc_sigma2_beta)
        post_sigma2_f = torch.mean(self.mcmc_sigma2_f)
        post_sigma2_u = torch.mean(self.mcmc_sigma2_u )
        # return post_thresh_alpha, post_theta_f, post_theta_alpha, post_zeta, post_beta, post_theta_nn, post_f, post_u, post_sigma2_u, post_sigma2_f, post_sigma2_beta, post_sigma2_nn 
        return post_alpha, post_delta, post_beta, post_zeta, post_alpha_prime, post_u, post_sigma2_u, post_sigma2_f, post_sigma2_beta, post_sigma2_nn 


    def post_CI(self):
        f_lci = torch.quantile(self.mcmc_f, 0.025, 0)
        f_uci = torch.quantile(self.mcmc_f, 0.975, 0)
        f_mean = torch.mean(self.mcmc_f, 0)
        return f_lci, f_uci, f_mean
    
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


def post_loglik(thresh_alpha, theta_alpha, theta_f, theta_nn, beta, zeta, U, U_mean, y, X, Phi, lam, sigma_f, sigma_u, sigma_nn,sigma_beta):
    log_lik_theta_f = 0
    log_lik_theta_nn = 0
    log_lik_theta_alpha = 0
    log_lik_beta = 0
    
    L = Phi.shape[1]
    N = X.shape[0]
    J = X.shape[1]
    V = Phi.shape[0]
    Q_V = Phi.t() @ thresh_alpha / V ### L x 1
    temp = U - U_mean
    sigma2_theta = 1/(Q_V **2 / sigma_u**2 + lam / sigma_f**2 + 1)
    for l in range(L): 
        temp += Q_V[l] * theta_f[l,]
        mu_theta = sigma2_theta[l] * (temp * Q_V[l] / sigma_u**2  + (Phi.t() @ y)[l,]/ sigma_f**2)
        dist = torch.distributions.Normal(mu_theta, torch.sqrt(sigma2_theta[l]))
        log_lik_theta_f += torch.sum(dist.log_prob(theta_f[l,]))
        temp -= Q_V[l] * theta_f[l,]

    log_prior = - 0.5 * torch.sum(theta_nn ** 2)/ sigma_nn**2
    log_ll = - 0.5 * torch.sum((y - Phi @ theta_f) ** 2) / sigma_f**2 
    alpha = Phi @ theta_alpha
    thresh_alpha = torch.where(torch.abs(alpha) > zeta, alpha, 0)
    u_est_batch_temp = (Phi @ theta_f).t() @ thresh_alpha/V + X @ beta
    rss_u_temp = torch.sum(( - u_est_batch_temp) ** 2)
    log_ll += - 0.5 * rss_u_temp / sigma_u**2
    log_lik_theta_nn = (log_prior + log_ll)/N

    temp = U - U_mean
    lam_theta_V = torch.mm(torch.diag(lam), theta_f) / V
    sigma2_alpha = 1 / (torch.sum(theta_f ** 2, 1) / sigma_u**2 * (lam / V) ** 2 + 1)
    for l in range(L):
        temp += lam_theta_V[l,] * theta_alpha[l]
        mu_alpha = sigma2_alpha[l] * torch.sum(temp * lam_theta_V[l,]) / sigma_u**2
        dist = torch.distributions.Normal(mu_alpha, torch.sqrt(sigma2_alpha[l]))
        log_lik_theta_alpha += dist.log_prob(theta_alpha[l])
        temp -= lam_theta_V[l,] * theta_alpha[l]

    temp = U - U_mean
    sigma2_beta = 1 / (torch.sum(X ** 2, 0)/sigma_u**2 + 1/sigma_beta**2)
    for j in range(J):
        temp += X[:,j] * beta[j]
        mu_beta = sigma2_beta[j] * torch.sum(temp * X[:,j]) / sigma_u**2
        dist = torch.distributions.Normal(mu_beta, torch.sqrt(sigma2_beta[j]))
        beta[j] = dist.sample()
        log_lik_beta += dist.log_prob(beta[j])
        temp -= X[:,j] * beta[j]

    return log_lik_theta_nn, log_lik_theta_alpha, log_lik_theta_f, log_lik_beta



def get_weights(m):
    param = torch.empty(0)
    for p in m.parameters():
      p_vec = p.reshape(-1)
      param = torch.cat((param, p_vec))
    
    return param

def return_weights(m, param, cumsum_vars):
    for i, p in enumerate(m.parameters()):
        #value = torch.tensor(param[cumsum_vars[i]:cumsum_vars[i+1]], dtype = torch.float32).reshape(p.shape)
        #p.data = value
        value = (param[cumsum_vars[i]:cumsum_vars[i+1]]).clone().detach().requires_grad_(True).reshape(p.shape)
        p.data = value
    return(m)

def generate_grids(img_shape, V_range=[-1,1]):
    d = len(img_shape)
    V = np.prod(img_shape)
    assert((d >= 1) & (d <=3))
    if (d == 1) :
          x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
          S = torch.from_numpy(x_).reshape(V,1)
    elif (d == 2) :
            x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
            y_ = np.linspace(V_range[0],  V_range[1], img_shape[1])
            x, y = np.meshgrid(x_, y_,  indexing='ij')
            S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1))))
    else:
            x_ = np.linspace(V_range[0],  V_range[1], img_shape[0])
            y_ = np.linspace(V_range[0],  V_range[1], img_shape[1])
            z_ = np.linspace(V_range[0],  V_range[1], img_shape[2])

            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1))))
    return S


def generate_basis(grids, L, model, sigma_nn):
    V = grids.shape[0]
    d = grids.shape[1]

    #get index for NN vec parameters
    theta_nn_cumsum_ind = np.cumsum([p.numel() for p in model.parameters() if p.requires_grad])
    theta_nn_cumsum_ind = np.concatenate(([0], theta_nn_cumsum_ind))
    #put true weights and bias in NN
    theta_nn = get_weights(model)
    theta_nn = torch.randn_like(theta_nn) * sigma_nn
    model = return_weights(model, theta_nn, theta_nn_cumsum_ind)

    nn_output = model(grids)
    Psi, lamb, Vh = torch.linalg.svd(nn_output,  full_matrices=False)
    #U.shape;E.shape;Vh.shape
    for l in range(L) :
            if(Psi[0,l] < 0):
                Psi[:,l] = Psi[:,l] *(-1)
    Lam_sqrt = torch.diag(torch.sqrt(lamb))
    Phi = torch.linalg.matmul(Psi, Lam_sqrt).detach()

    return model, Phi, theta_nn, theta_nn_cumsum_ind

def generate_data(Phi, N, J, sigma_f, sigma_u, sigma_alpha, zeta=None):
    V = Phi.shape[0]
    L = Phi.shape[1]


    #covariates from std normal
    X = torch.randn(N, J)
    
    #initialization
    theta_f = torch.randn(L, N) 
    theta_beta = torch.randn(L)  
    alpha = torch.randn(J) * sigma_alpha
    beta = Phi @ theta_beta
    if zeta is None:
        kth = round(V/2)
        zeta, ind = torch.kthvalue(torch.abs(beta), kth, 0, True)
        zeta = zeta.view(-1)

    zeta_mat = torch.from_numpy(np.full_like(beta.numpy(), zeta))
    thresh_beta = torch.where(torch.abs(beta) > zeta_mat, beta, 0)

    ### generate image
    eps_f = torch.randn(V, N) * sigma_f
    f = Phi @ theta_f
    Y = f + eps_f
    
    ### generate scalar outcome
    eps_u = torch.randn(N) * sigma_u ## noise
    U = (Y.t() @ thresh_beta) + X @ alpha + eps_u

    return theta_f, theta_beta, thresh_beta, zeta, alpha, f, X, Y, U