import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from models.nalu import NALU
from models.inalu import iNALU
from models.nac import NAC
from models.gnalu import GNALU
from models.nau import NAU
from models.nmu import NMU
from models.npu import NPU
from models.realnpu import RealNPU
from models.inpu import iNPU

from training.utils import *
# =============================================================================
# Hàm Benchmark Core
# =============================================================================

def get_model_class(model_name):
    '''Mapping tên model từ string sang class'''
    model_name = model_name.upper()
    mapping = {
        'NALU': NALU,
        'INALU': iNALU,
        'NAC': NAC,
        'GNALU': GNALU,
        'NAU': NAU,
        'NMU': NMU,
        'NPU': NPU,
        'REALNPU': RealNPU,
        'INPU': iNPU,
    }
    if model_name not in mapping:
        raise ValueError(f"Model {model_name} chưa được định nghĩa trong mapping.")
    return mapping[model_name]

def run_benchmark(args):
    results = []
    device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...")

    # Setup param
    modelClass = get_model_class(args.model)
    op_hash = {
        'add': (2, 6),
        'sub': (3, 7),
        'mul': (4, 8),
        'div': (5, 9)
    }
    if args.rangeId is None:
        rangeIDs = [f'U{i}' for i in range(1, 10)]
    else:
        rangeIDs = [rid.strip() for rid in list(args.rangeId.split(','))]

    # Hyperparameter
    lambda_base, lambda_start, lambda_end = 0, 20000, 35000
    beta_start, beta_end, beta_growth, beta_step = 0, 0, 10, 10000
    lr = 1e-3
    batch_size = args.batch_size
    n_iterations = args.n_iterations
    log_interval = 1000
    verbose = True

    if isinstance(modelClass, NAU): 
        lambda_base = 0.01
    elif isinstance(modelClass, NMU): 
        lambda_base = 10
    elif isinstance(modelClass, (NPU, RealNPU)):
        lr = 5e-3
        if args.op == 'mul':
            beta_start, beta_end = 1e-7, 1e-5
        elif args.op == 'div':
            beta_start, beta_end = 1e-9, 1e-7

    # Training
    for rid in tqdm(rangeIDs, desc='Ranges'):
        data_val = torch.tensor(np.load(f'handle_data/data/range_{rid}_val.npz')['data'], device=device)
        data_test = torch.tensor(np.load(f'handle_data/data/range_{rid}_test.npz')['data'], device=device)

        X_val, Y_val = data_val[:, :2], data_val[:, op_hash[args.op][0]].unsqueeze(1)
        X_test, Y_test = data_test[:, :2], data_test[:, op_hash[args.op][0]].unsqueeze(1)

        mse_val = F.mse_loss(data_val[:, op_hash[args.op][0]], data_val[:, op_hash[args.op][1]])
        mse_test = F.mse_loss(data_test[:, op_hash[args.op][0]], data_test[:, op_hash[args.op][1]])

        print()
        print(f'MSE_val: {mse_val} | MSE_test: {mse_test}')

        # Training
        for _ in tqdm(range(25), desc='Seeds'):
            print('\n\n')
            # Reinit model
            model = modelClass(in_dim=2, out_dim=1, device=device)

            # Generate new training data
            X_train, Y_train = rand_data_train(n_iterations=n_iterations, batch_size=batch_size,
                                               op=args.op, rid=rid)
            X_train = X_train.to(device=device)
            Y_train = Y_train.to(device=device)

            # Fit model
            history = {
                'interpolation_loss': [],
                'extrapolation_loss': [],
                'sparsity_loss'     : []
            }
            if args.optimizer == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif args.optimizer == 'SGD': optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else: raise ValueError(f'Unknown Optimization Algorithm: {args.optimizer}\n')

            for iter in range(1, n_iterations + 1):
                X = X_train[iter*batch_size : (iter + 1)*batch_size]
                Y = Y_train[iter*batch_size : (iter + 1)*batch_size]

                model.train()
                lambda_current = lambda_base * min(1.0, max(0.0, (iter - lambda_start) / (lambda_end - lambda_start)))
                beta_current   = min(beta_start * (beta_growth ** (iter // beta_step)), beta_end)
                
                total_loss = (F.mse_loss(model(X), Y) + 
                              lambda_current * model.regularization_loss() + 
                              beta_current * model.regularization_loss())
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if iter % log_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        history['interpolation_loss'].append(F.mse_loss(model(X_val), Y_val))
                        history['extrapolation_loss'].append(F.mse_loss(model(X_test), Y_test))
                        history['sparsity_loss'].append(calc_sparsity_loss(model, device))

                        if verbose:
                            print(f'Interation: {iter} | Train Loss: {total_loss} | ',
                                  f'Validation Loss: {history['interpolation_loss'][-1]} | ',
                                  f'Extrapolation Loss: {history['extrapolation_loss'][-1]}')
                
            # Post-process
