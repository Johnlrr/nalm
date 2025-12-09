import torch
import torch.nn as nn

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
        self.G = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        self.to(device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.G)
        self.G.data /= 2

        nn.init.xavier_uniform_(self.W_real)

    def regularization_loss(self):
        return torch.sum(torch.abs(self.W_real) + torch.abs(self.G))

    def forward(self, x):
        x = x.to(self.device)

        G = torch.clamp(self.G, 0.0, 1.0)
        R = torch.einsum('ni,io->nio', torch.abs(x) + self.epsilon, G) + (1 - G).unsqueeze(0)
        K = torch.pi * torch.einsum('ni,io->nio', (x < 0).float(), G)

        C = torch.einsum('nio,io->no', torch.log(R), self.W_real)
        D = torch.einsum('nio,io->no', K, self.W_real)

        return torch.exp(C) * torch.cos(D)