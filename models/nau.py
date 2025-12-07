import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NAU(nn.Module):
    """
    Neural Addition Unit (NAU)
    Paper: Neural Arithmetic Units (Madsen et al., 2020)
    
    Architecture:
    y = x * W^T
    
    Điểm khác biệt so với Linear layer thường:
    - Không có bias.
    - Không dùng activation function (như tanh/sigmoid của NAC).
    - Phụ thuộc hoàn toàn vào Sparsity Regularization trong quá trình training 
      để ép W về {-1, 0, 1}.
    """
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Ma trận trọng số W
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        
        self.reset_parameters()
        self.to(device=self.device)

    def reset_parameters(self):
        # Khởi tạo Xavier Uniform (Glorot)
        # Giúp ổn định variance của gradient
        nn.init.xavier_uniform_(self.W)

    def regularization_loss(self):
        # Regularization để ép W về {-1, 0, 1}
        W_abs = torch.abs(self.W)
        return torch.mean(torch.minimum(W_abs, 1 - W_abs))

    def forward(self, x):
        x = x.to(self.device)
        # NAU thực chất là phép nhân ma trận tuyến tính
        # Regularization sẽ được tính bên ngoài vòng lặp training
        return torch.matmul(x, self.W)

    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            optimizer_algo='Adam', threshold=1e-5,
            batch_size=128,
            lambda_base=0.01, lambda_start=20000, lambda_end=35000):
        
        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)
        Y_train = Y_train.to(self.device)
        Y_test = Y_test.to(self.device)

        if optimizer_algo == 'Adam': optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_algo == 'SGD': optimizer = optim.SGD(self.parameters(), lr=lr)
        else: raise ValueError(f'Unknown Optimization Algorithm: {optimizer_algo}\n')
        
        batch = X_train.shape[0]

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