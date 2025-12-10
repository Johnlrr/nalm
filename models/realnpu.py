import torch
import torch.nn as nn
from training.utils import calc_sparsity_loss

class RealNPU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-36):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.W_real = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.G = nn.Parameter(torch.Tensor(in_dim))
        
        self.to(device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.G)
        self.G.data /= 2

        nn.init.xavier_uniform_(self.W_real)

    def sparsity_loss(self):
        W_real = torch.clamp(self.W_real, 0.0, 1.0)
        G = torch.clamp(self.G, 0.0, 1.0)
        return torch.max(calc_sparsity_loss(W_real), calc_sparsity_loss(G))

    def regularization_loss(self):
        return torch.sum(torch.abs(self.W_real) + torch.abs(self.G))

    def forward(self, x):
        x = x.to(self.device)

        G = torch.clamp(self.G, 0.0, 1.0)
        R = torch.einsum('ni,i->ni', torch.abs(x) + self.epsilon, G) + (1 - G).unsqueeze(0)
        K = torch.pi * torch.einsum('ni,i->ni', (x < 0).float(), G)

        C = torch.einsum('ni,io->no', torch.log(R), self.W_real)
        D = torch.einsum('ni,io->no', K, self.W_real)
        return torch.exp(C) * torch.cos(D)