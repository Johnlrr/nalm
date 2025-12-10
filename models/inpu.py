import torch
import torch.nn as nn
from training.utils import calc_sparsity_loss

class iNPU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-16):
        super().__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim
        self.epsilon = epsilon
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.W = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.to(device=self.device)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.W, a=-0.1, b=0.1)
        nn.init.zeros_(self.W)

    def sparsity_loss(self):
        # W = torch.clamp(self.W, min=0, max=1)
        return calc_sparsity_loss(self.W)

    def regularization_loss(self):
        # frac_W = torch.frac(torch.abs(self.W))
        # return torch.mean(torch.min(frac_W, 1 - frac_W))
        return 0

    def forward(self, X):
        X = X.to(self.device)

        magnitude = torch.exp(torch.matmul(torch.log(torch.clamp(torch.abs(X), min=self.epsilon)), self.W))
        sign = torch.cos(torch.pi * torch.matmul((X < 0).to(X.dtype), torch.round(self.W)))

        return sign * magnitude