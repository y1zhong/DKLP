import torch
from torch import nn
import pickle
import numpy as np

def log1pexp(u):
    val = torch.zeros_like(u)
    val += torch.where(u <= -37, torch.exp(u), 0)
    val += torch.where((u > -37 ) & (u <= 18), torch.log(1+torch.exp(u)), 0)
    val += torch.where((u > 18 ) & (u <= 33.3), u + torch.exp(-u), 0)
    val += torch.where((u >= 33.3 ) , u , 0)
    return val

def FDR_IS(prob, level, intercept = 0):
    V = prob.shape[0]
    J = prob.shape[1]
    pred = torch.ones(V,J)
    sort_p,ind = torch.sort(prob, dim=0,descending=True)
    lamb = torch.zeros(V)
    p_cutoff = torch.zeros(J)
    for j in range(J) :
        if (intercept == 1) and (j == 0):
            continue

        for v in range(V):
            lamb[v] = torch.sum(1 - sort_p[:v,j])/(v+1)
        ind = (lamb <= level).nonzero().flatten()
        if ind.nelement() == 0:
            p_lam = 0
        else:
            p_lam = sort_p[ind[-1],j]
        pred[:,j] = torch.where(prob[:,j] > p_lam, 1.0, 0.0)
        p_cutoff[j] = p_lam

    return pred


def FDR_SI(prob, level):
    V = prob.shape[0]
    pred = torch.ones(V)
    sort_p,ind = torch.sort(prob, dim=0,descending=True)
    lamb = torch.zeros(V)
    p_cutoff = 0

    for v in range(V):
        lamb[v] = torch.sum(1 - sort_p[:v])/(v+1)
    ind = (lamb <= level).nonzero().flatten()
    if ind.nelement() == 0:
        p_lam = 0
    else:
        p_lam = sort_p[ind[-1]]
    pred = torch.where(prob >= p_lam, 1.0, 0.0)
    p_cutoff = p_lam
    prob_cut = prob * pred
    return pred, p_cutoff, prob_cut


class DKLP_NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, M=4, act_fn='relu'):
        activation_dict = {
                'relu': nn.ReLU(inplace=True),
                'leaky': nn.LeakyReLU(0.1, inplace=True),
                'leaky0100': nn.LeakyReLU(0.100, inplace=True),
                'leaky0010': nn.LeakyReLU(0.010, inplace=True),
                'swish': nn.SiLU(inplace=True),
                'sigmoid': nn.Sigmoid(),
                }
        act = activation_dict[act_fn]
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)

        for _ in range(M):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)
    

def save_pickle(x, filename):
    # mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print('save',filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    print(f'Pickle loaded from {filename}')
    return x

def generate_grids(d, img_shape, V_range=[-1, 1]):
    #assert np.prod(img_shape) == V
    V = np.prod(img_shape)
    if (d == 1) :
            S = np.linspace(V_range[0], V_range[1], int(V))
            S = torch.from_numpy(S).reshape(V, 1)
    elif (d == 2) :
        x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
        y_ = np.linspace(V_range[0], V_range[1], img_shape[1])
        x, y = np.meshgrid(x_, y_, indexing='ij')
        S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1))))
    elif (d == 3) :
        x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
        y_ = np.linspace(V_range[0], V_range[1], img_shape[1])
        z_ = np.linspace(V_range[0], V_range[1], img_shape[2])

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1))))

    return S.float()