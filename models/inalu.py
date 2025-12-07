import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class iNALU(nn.Module):
    """
    Improved Neural Arithmetic Logic Unit (iNALU)
    Paper: iNALU: Improved Neural Arithmetic Logic Unit (Schlör et al., 2020)
    
    Cải tiến so với NALU gốc:
    1. Independent Weights: Tách biệt hoàn toàn trọng số cộng và nhân (đã làm ở NALU).
    2. Input Clipping: Kẹp giá trị trước khi vào log để tránh log(0) hoặc log(số quá nhỏ).
    3. Gradient Stability: Giới hạn giá trị mũ để tránh tràn số (overflow).
    """
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-36, omega=20, t=20):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.omega = omega
        self.t = t

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.W_hat_add = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.M_hat_add = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.W_hat_mul = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.M_hat_mul = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.G = nn.Parameter(torch.Tensor(1, out_dim))
        
        self.to(device=self.device)
        self.reset_parameters()

    def regularization_loss(self):
        reg_loss = 0.0
        reg_loss += torch.mean(torch.clamp(torch.minimum(-self.W_hat_add, self.W_hat_add) + self.t, min=0))
        reg_loss += torch.mean(torch.clamp(torch.minimum(-self.W_hat_mul, self.W_hat_mul) + self.t, min=0))
        reg_loss += torch.mean(torch.clamp(torch.minimum(-self.M_hat_add, self.M_hat_add) + self.t, min=0))
        reg_loss += torch.mean(torch.clamp(torch.minimum(-self.M_hat_mul, self.M_hat_mul) + self.t, min=0))
        return reg_loss / self.t

    def reset_parameters(self):
        # Khởi tạo Xavier
        nn.init.xavier_uniform_(self.W_hat_add)
        nn.init.xavier_uniform_(self.M_hat_add)
        nn.init.xavier_uniform_(self.W_hat_mul)
        nn.init.xavier_uniform_(self.M_hat_mul)
        nn.init.xavier_uniform_(self.G)

    def forward(self, x):
        x = x.to(self.device)

        W_add = torch.tanh(self.W_hat_add) * torch.sigmoid(self.M_hat_add)
        W_mul = torch.tanh(self.W_hat_mul) * torch.sigmoid(self.M_hat_mul)

        a = torch.matmul(x, W_add)

        log_x = torch.log(torch.clamp(torch.abs(x), min=self.epsilon))
        m_log = torch.clamp(torch.matmul(log_x, W_mul), max=self.omega)
        m = torch.exp(m_log)
        msv = torch.prod(torch.einsum('ni,io->nio', x - 1, W_mul) + 1, dim=1)

        g = torch.sigmoid(self.G)

        # Combine
        y = g * a + (1 - g) * msv * m
        return y

    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            optimizer_algo='Adam', threshold=1e-5):
        
        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)
        Y_train = Y_train.to(self.device)
        Y_test = Y_test.to(self.device)

        if optimizer_algo == 'Adam': optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_algo == 'SGD': optimizer = optim.SGD(self.parameters(), lr=lr)
        else: raise ValueError(f'Unknown Optimization Algorithm: {optimizer_algo}\n')
    
        for epoch in range(1, epochs + 1):
            self.train()
            
            Y_pred = self(X_train)
            train_loss = 0.5 * F.mse_loss(Y_pred, Y_train) + self.regularization_loss()

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