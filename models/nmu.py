import torch
import torch.nn as nn
import math
from training.utils import calc_sparsity_loss

class NMU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        self.reset_parameters()
        self.to(device=self.device)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

    def sparsity_loss(self):
        W = torch.clamp(self.W, 0.0, 1.0)
        return calc_sparsity_loss(W)

    def regularization_loss(self):
        W_abs = torch.abs(self.W)
        reg = torch.mean(torch.minimum(W_abs, 1 - W_abs))
        return reg

    def forward(self, x):
        x = x.to(self.device)
        
        W = torch.clamp(self.W, 0.0, 1.0)
        temp = torch.einsum('ni,io->nio', x - 1, W) + 1
        y = torch.prod(temp, dim=1)
        
        return y