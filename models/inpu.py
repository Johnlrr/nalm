import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from utils.data_utils import convert_to_tensor

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

    def forward(self, X):
        X = X.to(self.device)

        magnitude = torch.exp(torch.matmul(torch.log(torch.clamp(torch.abs(X), min=self.epsilon)), self.W))
        sign = torch.cos(torch.pi * torch.matmul((X < 0).to(X.dtype), self.W))

        return sign * magnitude
    
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

        log_Y_train = torch.log(torch.clamp(torch.abs(Y_train), min=self.epsilon))
        clamped_log_X_train = torch.clamp(torch.abs(X_train), min=self.epsilon)
        
        for epoch in range(1, epochs + 1):
            self.train()
            
            magnitude_log = torch.matmul(
                torch.log(clamped_log_X_train),
                self.W
            )
            train_loss = 0.5 * F.mse_loss(magnitude_log, log_Y_train)

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