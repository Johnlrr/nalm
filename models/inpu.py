import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.data_utils import convert_to_tensor

class iNPU(nn.Module):
    def __init__(self, input_dim, output_dim, epsilon=1e-36):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)

    def regularization(self):
        frac_W = torch.frac(torch.abs(self.W))
        return torch.mean(torch.minimum(frac_W, 1 - frac_W))

    def forward(self, X):
        magnitude = torch.exp(torch.matmul(torch.log(torch.clamp(torch.abs(X), min=self.epsilon)), self.W))
        sign = torch.cos(torch.pi * torch.matmul((X < 0).type(torch.float64), self.W))

        return sign * magnitude
    
    def fit(self, X_train, Y_train, X_test, Y_test, 
            lr=1e-3, epochs=50000, each_epoch=1000,
            optimizer_algo='Adam', threshold=1e-5):
        
        convert_to_tensor([X_train, Y_train, X_test, Y_test])

        if optimizer_algo == 'Adam': optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_algo == 'SGD': optimizer = optim.SGD(self.parameters(), lr=lr)
        else: raise ValueError(f'Unknown Optimization Algorithm: {optimizer_algo}\n')

        log_Y_train = torch.log(torch.abs(Y_train))
        sign_Y_train = torch.sign(Y_train)
        
        for epoch in range(1, epochs + 1):
            self.train()
            
            magnitude_log = torch.matmul(torch.log(torch.clamp(torch.abs(X_train), min=self.epsilon)), self.W)
            train_loss = 0.5 * F.mse_loss(magnitude_log, log_Y_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()

                test_loss = 0.5 * F.mse_loss(self(X_test), Y_test)

                if epoch % each_epoch == 0:
                    # print('\n Weight', self.W)
                    print(f'Epoch: {epoch} | Loss: {train_loss} | Extrapolation Loss: {test_loss}')
                
                if test_loss < threshold:
                    return epoch, True
        
        return epochs, False