import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import *
import numpy as np

# Cấu trúc 9x2 = 18 file .npz: range_{ID}_{split}.npz. Ví dụ range_U1_val.npz, range_U5_val.npz, ...
# Mỗi file là một matrix n_samples x 10 với 2 cột x1, x2 và 8 cột y_add,sub,mul,div, y_add_ep,...

if __name__ == "__main__":
    n_samples = 10000
    epsilon = 1e-5
    seed_val = 100
    seed_test = 200

    print("=" * 70)
    print("Generating NALM Benchmark Datasets")
    print(f"  n_samples: {n_samples}")
    print(f"  epsilon: {epsilon}")
    print(f"  seed_val: {seed_val}")
    print(f"  seed_test: {seed_test}")
    print("=" * 70)

    for rid in [f'U{i}' for i in range(1, 10)]:
        for split in ['val', 'test']:
            data = {}
            file_save = f'handle_data/data/range_{rid}_{split}.npz'

            if split == 'val':
                data = list(generate_data(op=['add', 'sub', 'mul', 'div'], range_cfg_id=rid, type='inter',
                                          n_samples=n_samples, seed=seed_val, epsilon=0))
                data.append(generate_data(op=['add', 'sub', 'mul', 'div'], range_cfg_id=rid, type='inter',
                                          n_samples=n_samples, seed=seed_val, epsilon=epsilon)[1])
            else:
                data = list(generate_data(op=['add', 'sub', 'mul', 'div'], range_cfg_id=rid, type='extra',
                                          n_samples=n_samples, seed=seed_test, epsilon=0))
                data.append(generate_data(op=['add', 'sub', 'mul', 'div'], range_cfg_id=rid, type='extra',
                                          n_samples=n_samples, seed=seed_test, epsilon=epsilon)[1])
            
            merge_data = torch.cat(list(data), dim=1)
            print(merge_data.shape)
            np.savez_compressed(file_save, data=merge_data.numpy())
            print(f'Done {file_save}')