import torch
import torch.nn as nn

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
        # see https://github.com/daschloer/inalu/blob/e41d80d3506ac0bf4a2971c262003468feca187d/nalu_architectures.py#L4
        mu_g = 0.0
        mu_m = 0.5
        mu_w = 0.88

        sd_g = 0.2
        sd_m = 0.2
        sd_w = 0.2

        torch.nn.init.normal_(self.W_hat_add, mean=mu_w, std=sd_w)
        torch.nn.init.normal_(self.M_hat_add, mean=mu_m, std=sd_m)
        torch.nn.init.normal_(self.W_hat_mul, mean=mu_w, std=sd_w)
        torch.nn.init.normal_(self.M_hat_mul, mean=mu_m, std=sd_m)
        torch.nn.init.normal_(self.G, mean=mu_g, std=sd_g)

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