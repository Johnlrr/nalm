import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NAC(nn.Module):
    """
    Neural Accumulator (NAC)
    Paper: Neural Arithmetic Logic Units (Trask et al., 2018)
    
    Logic:
    W = tanh(W_hat) * sigmoid(M_hat)
    y = x * W^T
    """
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Khởi tạo 2 ma trận tham số W_hat và M_hat
        self.W_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.M_hat = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.to(device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo Xavier Uniform (Glorot) được khuyến nghị trong paper benchmark
        # giúp gradient lan truyền tốt hơn qua tanh và sigmoid
        nn.init.xavier_uniform_(self.W_hat)
        nn.init.xavier_uniform_(self.M_hat)

    def forward(self, x):
        x = x.to(self.device)
        # Tính trọng số W ép về khoảng {-1, 0, 1}
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        
        # Thực hiện phép biến đổi tuyến tính: y = xW^T
        return torch.matmul(x, W)

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