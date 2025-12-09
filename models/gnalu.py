import torch
from models.nalu import NALU
from training.utils import calc_sparsity_loss

class GNALU(NALU):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-7):
        super().__init__(in_dim, out_dim, device, epsilon)
        
        self.phi = (1 + 5 ** 0.5) / 2  # Tỷ lệ vàng
        self.reset_parameters()
        self.to(device=self.device)

    def reset_parameters(self):
        super().reset_parameters()
    
    def golden_tanh(self, x):
        """Hàm tanh điều chỉnh theo tỷ lệ vàng"""
        x = x.to(self.device)
        phi_2x = torch.pow(self.phi, 2 * x)
        return (phi_2x - 1)/(phi_2x + 1)

    def golden_sigmoid(self, x):
        """Hàm sigmoid điều chỉnh theo tỷ lệ vàng"""
        x = x.to(self.device)

        return 1 / (torch.pow(self.phi, -x) + 1)

    def sparsity_loss(self):
        W = self.golden_tanh(self.W_hat) * self.golden_sigmoid(self.M_hat)
        G = torch.sigmoid(self.G)
        return torch.max(calc_sparsity_loss(W), calc_sparsity_loss(G))

    def regularization_loss(self):
        return 0

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