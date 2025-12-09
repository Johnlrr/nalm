import torch
import torch.nn as nn

class iNPU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-36):
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
        nn.init.xavier_uniform_(self.W)

    def regularization_loss(self):
        return 0

    def forward(self, X):
        X = X.to(self.device)

        magnitude = torch.exp(torch.matmul(torch.log(torch.clamp(torch.abs(X), min=self.epsilon)), self.W))
        sign = torch.cos(torch.pi * torch.matmul((X < 0).to(X.dtype), self.W))

        return sign * magnitude