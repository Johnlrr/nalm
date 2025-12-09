import torch
import torch.nn as nn
import math
from training.utils import calc_sparsity_loss

class NAU(nn.Module):
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
        std = math.sqrt(2.0 / (self.in_dim + self.out_dim))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def sparsity_loss(self):
        W = torch.clamp(self.W, -1, 1)

        return calc_sparsity_loss(W)

    def regularization_loss(self):
        W_abs = torch.abs(self.W)
        return torch.mean(torch.minimum(W_abs, 1 - W_abs))

    def forward(self, x):
        W = torch.clamp(self.W, -1, 1)
        return torch.matmul(x, W)