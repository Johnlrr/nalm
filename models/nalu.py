import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NALU(nn.Module):
    """
    Neural Arithmetic Logic Unit (NALU)
    Paper: Neural Arithmetic Logic Units (Trask et al., 2018)
    
    Architecture:
    1. Arithmetic Path (a): NAC(x) -> Học cộng/trừ
    2. Multiplicative Path (m): exp(NAC(log(|x| + eps))) -> Học nhân/chia
    3. Gate (g): sigmoid(G * x) -> Chọn giữa (a) và (m)
    
    Output: y = g * a + (1 - g) * m
    """
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
        # Khởi tạo tham số cho Gate
        nn.init.xavier_uniform_(self.G)
        # Khởi tạo tham số cho NAC cộng/trừ
        nn.init.xavier_uniform_(self.W_hat)
        nn.init.xavier_uniform_(self.M_hat)

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

    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            batch_size=128,
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
            train_loss = 0.5 * F.mse_loss(Y_pred, Y_train)

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