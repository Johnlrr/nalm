import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NMU(nn.Module):
    """
    Neural Multiplication Unit (NMU)
    Paper: Neural Arithmetic Units (Madsen et al., 2020)
    
    Logic:
    y = Product_over_inputs(W * x + (1 - W))
    
    Yêu cầu: 
    - W phải nằm trong khoảng [0, 1].
    - W đóng vai trò như một "cổng mềm" (soft gate):
        + W -> 1: Chọn input đó để nhân.
        + W -> 0: Bỏ qua input đó (nhân với 1).
    """
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
        # Khởi tạo trọng số trong khoảng [0, 0.5] hoặc [0, 1]
        # Madsen et al. khuyến nghị khởi tạo uniform quanh 0.5 hoặc nhỏ hơn 
        # để bắt đầu "trung lập".
        nn.init.uniform_(self.W, 0, 0.5)

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

    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            optimizer_algo='Adam', threshold=1e-5,
            batch_size=128,
            lambda_base=10, lambda_start=20000, lambda_end=35000):
        
        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)
        Y_train = Y_train.to(self.device)
        Y_test = Y_test.to(self.device)
        
        if optimizer_algo == 'Adam': optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_algo == 'SGD': optimizer = optim.SGD(self.parameters(), lr=lr)
        else: raise ValueError(f'Unknown Optimization Algorithm: {optimizer_algo}\n')

        for epoch in range(1, epochs + 1):
            self.train()
            lambda_current = lambda_base * min(1.0, max(0.0, (epoch - lambda_start) / (lambda_end - lambda_start)))

            Y_pred = self(X_train)
            train_loss = 0.5 * F.mse_loss(Y_pred, Y_train) + lambda_current * self.regularization_loss()

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