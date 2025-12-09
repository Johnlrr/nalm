import torch
import numpy as np

# =================================================
# Generate Data cho từng phép toán và range config
# =================================================

RANGE_CONFIGS = {
    'U1': ([-20, -10], [[-40, -20]]),
    'U2': ([-2, -1], [[-6, -2]]),
    'U3': ([-1.2, -1.1], [[-6.1, -1.2]]),
    'U4': ([-0.2, -0.1], [[-2, -0.2]]),
    'U5': ([-2, 2], [[-6, -2], [2, 6]]), # TH đặc biệt bị rời
    'U6': ([0.1, 0.2], [[0.2, 2]]),
    'U7': ([1, 2], [[2, 6]]),
    'U8': ([1.1, 1.2], [[1.2, 6]]),
    'U9': ([10, 20], [[20, 40]])
}

def apply_op(op, x1, x2, epsilon):
    if op == 'add':
        return x1 + x2 - epsilon * (abs(x1) + abs(x2))
    elif op == 'sub':
        return x1 - x2 - epsilon * (abs(x1) + abs(x2))
    elif op == 'mul':
        return (x1 * x2) * (1 - epsilon)**2
    elif op == 'div':
        assert torch.all(x2 != 0)
        return x1 / x2 * (1 - epsilon) / (1 + epsilon)
    else:
        raise ValueError(f"Unknown op: {op}")
    
def generate_data(op, range_cfg_id, type='inter', n_samples=10000, seed=0, epsilon=0):
    """
    Truyền vào range_cfg là [key]: 'U1', 'U2',...
    Truyền vào op là mảng gồm add, sub, mul, div.
    Truyền vào type là inter (interpolation) hoặc extra (extrapolation)

    Trả về X, Y
    """
    # Đặt seed để tái lập kết quả
    torch.manual_seed(seed)
    
    if type == 'inter':
        train_range = RANGE_CONFIGS[range_cfg_id][0]
        low, high = train_range
        X = torch.zeros(n_samples, 2).uniform_(low, high)
    elif type == 'extra':
        test_range = RANGE_CONFIGS[range_cfg_id][1]
        num_ranges = len(test_range)
        per_range = n_samples // num_ranges
        
        parts = []
        current_count = 0

        for i, (low, high) in enumerate(test_range):
            count = per_range if i < num_ranges - 1 else n_samples - current_count
            
            part = torch.zeros(count, 2).uniform_(low, high)
            parts.append(part)
            current_count += count
            
        X = torch.cat(parts, dim=0)
    else:
        raise ValueError(f'type {type} is not defined')
    
    if isinstance(op, (list, tuple)):
        temp = []
        for operator in op:
            temp.append(apply_op(operator, X[:, 0], X[:, 1], epsilon).unsqueeze(1))
        Y = torch.cat(temp, dim=1)
    else:
        Y = apply_op(op, X[:, 0], X[:, 1], epsilon).unsqueeze(1)

    return X, Y

def rand_data_train(n_iterations, batch_size, op, rid, dim=2):
    low, high = RANGE_CONFIGS[rid][0]
    data = []
    for _ in range(n_iterations):
        torch.manual_seed(np.random.randint(0, (1<<31) - 1))
        data.append(torch.zeros(batch_size, dim).uniform_(low, high))
    
    X = torch.cat(data, dim=0)
    Y = apply_op(op, X[:, 0], X[:, 1], epsilon=0).unsqueeze(1)

    return X, Y

# =================================================
# helper cho tác vụ huấn luyện và đánh giá
# =================================================

def calc_sparsity_loss(model, device):
    '''Tính sparsity loss cho tất cả tham số trong model'''
    loss = torch.tensor(0.0, device=device) 
    for param in model.parameters():
        param_abs = torch.abs(param)
        loss = torch.max(loss, torch.max(torch.minimum(param_abs, torch.abs(1 - param_abs))))

    return loss