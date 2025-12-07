import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NALM Benchmark Runner")
    
    # Required args
    parser.add_argument("--model", type=str, required=True, help="Tên module (NAU, NMU, ...)")
    parser.add_argument("--op", type=str, required=True, choices=['add', 'sub', 'mul', 'div'], help="Toán tử")
    
    # Optional args with defaults
    parser.add_argument("--n_epochs", type=int, default=50000, help="Số lượng training epochs")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=['Adam', 'SGD'], help="Thuật toán tối ưu")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="Thiết bị chạy")
    parser.add_argument("--output", type=str, default="results/benchmark_results.csv", help="File output csv")

    args = parser.parse_args()

    # Chạy benchmark
    df = run_benchmark(args)
    
    # Hiển thị và Lưu
    print("\nBenchmark Completed!")
    print(df[['Range', 'Success_Rate', 'Speed_Convergence_Mean', 'Sparsity_Error_Mean']])
    
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")