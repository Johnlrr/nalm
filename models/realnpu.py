import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RealNPU(nn.Module):
    """
    Real Neural Power Unit (RealNPU)
    Paper: Neural Power Units (Heim et al., 2020)
    
    Logic:
    Mô phỏng phép nhân lũy thừa y = product(x_i ^ w_i) cho cả số âm.
    Tách biệt xử lý Độ lớn (Magnitude) và Dấu (Sign).
    
    Công thức:
    Magnitude = exp(W * log(|x| + eps))
    Sign      = cos(pi * W * (x < 0))
    Output    = Magnitude * Sign
    """
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
        # Xavier Uniform
        nn.init.xavier_uniform_(self.W_real)
        nn.init.uniform_(self.G, 0.0, 1.0)

    def forward(self, x):
        x = x.to(self.device)

        G = torch.clamp(self.G, 0.0, 1.0)
        R = torch.einsum('ni,io->nio', torch.abs(x) + self.epsilon, G) + (1 - G).unsqueeze(0)
        K = torch.pi * torch.einsum('ni,io->nio', (x < 0).float(), G)

        C = torch.einsum('nio,io->no', torch.log(R), self.W_real)
        D = torch.einsum('nio,io->no', K, self.W_real)

        return torch.exp(C) * torch.cos(D)
    
    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            optimizer_algo='Adam', threshold=1e-5,
            batch_size=128,
            beta_begin=1e-7, beta_end=1e-5, beta_growth=10, beta_step=10000):
        
        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)
        Y_train = Y_train.to(self.device)
        Y_test = Y_test.to(self.device)

        if optimizer_algo == 'Adam': optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_algo == 'SGD': optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else: raise ValueError(f'Unknown Optimization Algorithm: {optimizer_algo}\n')

        for epoch in range(1, epochs + 1):
            self.train()
            beta_current = min(beta_begin * (beta_growth ** (epoch // beta_step)), beta_end)

            Y_pred = self(X_train)
            train_loss = 0.5 * F.mse_loss(Y_pred, Y_train) + beta_current * torch.sum(torch.abs(self.W_real) + torch.abs(self.G))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                
                test_loss = 0.5 * F.mse_loss(self(X_test), Y_test)

                if each_epoch is not None and epoch % each_epoch == 0:
                    print(f'Epoch: {epoch} | Loss: {train_loss} | Extrapolation Loss: {test_loss}')
                
                if test_loss < threshold:
                    return epoch, True
        
        return epochs, False