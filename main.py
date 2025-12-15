import argparse
from training.training import *
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục training/
PROJECT_DIR = os.path.dirname(ROOT_DIR)                # thư mục nalm/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NALM Benchmark Runner")
    
    # Required args
    parser.add_argument("--model", type=str, required=True, help="Tên module (NAU, NMU, ...)")
    parser.add_argument("--op", type=str, required=True, choices=['add', 'sub', 'mul', 'div'], help="Toán tử")
    
    # Optional args with defaults
    parser.add_argument("--n_iterations", type=int, default=50000, help="Số lượng training iterations")
    parser.add_argument("--rangeID", type=str, default=None, help="Range ID: U1, U2, ...")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=['Adam', 'SGD'], help="Thuật toán tối ưu")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_seeds", type=int, default=25, help="Số lượng seeds khi benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="Thiết bị chạy")
    parser.add_argument("--verbose", action="store_true", help="In ra log khi chạy")
    parser.add_argument("--log_interval", type=int, default=1000, help='Khoảng cách giữa các lần đo lường')
    parser.add_argument("--output", type=str, default="results/benchmark_results.csv", help="File output csv")

    args = parser.parse_args()
    # Chạy benchmark
    df = run_benchmark(args)
    
    # Hiển thị và Lưu
    print("\nBenchmark Completed!")
    print(df[['Range', 'Success_Rate', 'Speed_Convergence_Mean', 'Sparsity_Error_Mean']])
    
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
