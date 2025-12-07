import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.nalu import NALU


class GNALU(NALU):
    """
    Golden Ratio NALU (GNALU)
    Paper: The Golden Ratio in the Initialization of Neural Arithmetic Logic Units
    
    Logic:
    Hoạt động y hệt NALU, nhưng thay đổi cách khởi tạo trọng số (Initialization).
    Sử dụng phân phối dựa trên tỷ lệ vàng (Golden Ratio) để tránh Dead Units.
    """
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-7):
        # Gọi init của NALU, nó sẽ tự gọi reset_parameters của class con (GNALU)
        super().__init__(in_dim, out_dim, device, epsilon)
        
        self.phi = (1 + 5 ** 0.5) / 2  # Tỷ lệ vàng
        self.reset_parameters()
        self.to(device=self.device)

    def reset_parameters(self):
        """Khởi tạo tham số theo chuẩn Golden Ratio"""
        std = (2 / (self.in_dim + self.out_dim)) ** 0.5 / ((1 + 5 ** 0.5) / 2)

        # Khởi tạo W_hat và M_hat theo phân phối chuẩn với độ lệch chuẩn điều chỉnh bởi tỷ lệ vàng
        self._init_nac_golden(std)
        
        # Khởi tạo tham số cho Gate
        nn.init.xavier_uniform_(self.G)

    def _init_nac_golden(self, std):
        nn.init.normal_(self.W_hat, mean=0.0, std=std)
        nn.init.normal_(self.M_hat, mean=0.0, std=std)

    def golden_tanh(self, x):
        """Hàm tanh điều chỉnh theo tỷ lệ vàng"""
        x = x.to(self.device)
        phi_2x = torch.pow(self.phi, 2 * x)
        return (phi_2x - 1)/(phi_2x + 1)

    def golden_sigmoid(self, x):
        """Hàm sigmoid điều chỉnh theo tỷ lệ vàng"""
        x = x.to(self.device)

        return 1 / (torch.pow(self.phi, -x) + 1)

    def forward(self, x):
        x = x.to(self.device)

        W = self.golden_tanh(self.W_hat) * self.golden_sigmoid(self.M_hat)

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