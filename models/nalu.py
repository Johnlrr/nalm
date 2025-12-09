import torch
import torch.nn as nn
import numpy as np
import scipy.optimize

# =================================================
# Khởi tạo trọng số trong NALU, G-NALU, NAC
# =================================================

def nac_w_variance(r):
    """Calculates the variance of W.

    Asumming \hat{w} and \hat{m} are sampled from a uniform
    distribution with range [-r, r], this is the variance
    of w = tanh(\hat{w})*sigmoid(\hat{m}).
    """
    if (r == 0):
        return 0
    else:
        return (1 - np.tanh(r) / r) * (r - np.tanh(r / 2)) * (1 / (2 * r))

def nac_w_optimal_r(fan_in, fan_out):
    """Computes the optimal Uniform[-r, r] given the fan

    This uses numerical optimization.
    TODO: consider if there is an algebraic solution.
    """
    fan = max(fan_in + fan_out, 5)
    r = scipy.optimize.bisect(lambda r: fan * nac_w_variance(r) - 2, 0, 10)
    return r

# =================================================
# CLASS NALU
# =================================================

class NALU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.W_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.M_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.G = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        self.to(device=self.device) 
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.G, gain=nn.init.calculate_gain('sigmoid'))

        r = nac_w_optimal_r(self.in_dim, self.out_dim)
        torch.nn.init.uniform_(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform_(self.M_hat, a=-r, b=r)

    def regularization_loss(self):
        return 0

    def forward(self, x):
        x = x.to(self.device)

        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        
        a = torch.matmul(x, W)

        x_abs = torch.abs(x)
        log_x = torch.log(x_abs + self.epsilon)
        m_log = torch.matmul(log_x, W)
        m = torch.exp(m_log)

        g = torch.sigmoid(torch.matmul(x, self.G))

        y = g * a + (1 - g) * m
        
        return y