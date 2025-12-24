import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import causal_conv1d_mps as ccmps


# Global dtype and tolerance (default bf16)
DTYPE = torch.bfloat16
ATOL_BY_DTYPE = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-1,
}


def get_tolerance_for(dtype: torch.dtype) -> float:
    return ATOL_BY_DTYPE.get(dtype, 1e-4)


def bench_robust(fn, warmup=10, iters=50, runs=5):
    """More stable performance test function"""
    results = []

    for run in range(runs):
        # Warmup before each run
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()

        # Measure multiple iterations
        times = []
        for _ in range(iters):
            t0 = time.time()
            fn()
            torch.mps.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

        # Remove highest and lowest values, calculate average
        times = sorted(times)[2:-2]  # Remove 2 extreme values from both ends
        avg_time = sum(times) / len(times)
        results.append(avg_time)

    # Remove highest and lowest runs, return median
    results = sorted(results)[1:-1]
    return sum(results) / len(results)


def bench_robust_stable(fn, warmup=25, iters=100, runs=5, desc=""):
    """
    A more stable and robust performance test function.

    Main improvements:
    1. Warmup before all runs.
    2. Use torch.mps.synchronize() for each measurement.
    3. Return median time of multiple runs, which is less sensitive to outliers.
    4. Return standard deviation for stability evaluation.
    """
    # Centralized warmup: before all timing starts, let GPU reach stable working state
    if desc:
        print(f"{desc:<20} {'Warming up...':<10}", end="\r")
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    run_times = []
    for _ in range(runs):
        # Each run is timed independently, more resistant to system interference
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        t1 = time.time()

        # Calculate average time per iteration
        avg_iter_time = (t1 - t0) / iters
        run_times.append(avg_iter_time)

    # Statistical analysis: calculate median and standard deviation
    median_time = np.median(run_times)
    std_dev = np.std(run_times)

    return median_time, std_dev


def causal_conv1d_reference(x, weight, bias=None, silu_activation=False):
    """PyTorch reference implementation"""
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # Use F.conv1d to implement causal convolution
    x_padded = F.pad(x, (width - 1, 0))  # Left padding

    out = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=dim, padding=0)
    out = out[:, :, :seqlen]  # Crop to original length

    if silu_activation:
        out = F.silu(out)

    return out


def main():
    print("🚀 Causal Conv1D MPS performance test")

    assert torch.backends.mps.is_available(), "MPS not available"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CausalConv1D MPS Benchmarks")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Test data type (default bf16)",
    )
    args = parser.parse_args()

    # Set global DTYPE and tolerance based on parameters
    global DTYPE
    if args.dtype == "bf16":
        DTYPE = torch.bfloat16
    elif args.dtype == "fp16":
        DTYPE = torch.float16
    else:
        DTYPE = torch.float32

    device = torch.device("mps")

    # Test configs: (batch, dim, seqlen, width)
    test_configs = [
        (1, 64, 128, 4),  # Small scale
        (2, 128, 256, 4),  # Medium scale
        (4, 256, 512, 4),  # Large scale
        (1, 512, 1024, 4),  # Huge scale
        (8, 64, 128, 4),  # Huge batch
    ]

    print(
        f"{'Config':<20} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 80)

    for batch, dim, seqlen, width in test_configs:
        # Create test data
        torch.manual_seed(42)
        x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
        weight = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias = torch.randn(dim, device=device, dtype=DTYPE)

        config_str = f"{batch}×{dim}×{seqlen}×{width}"

        try:
            # MPS implementation
            def run_mps():
                return ccmps.causal_conv1d_fwd(
                    x.contiguous(), weight.contiguous(), bias.contiguous(), False
                )

            # PyTorch reference implementation
            def run_torch():
                return causal_conv1d_reference(x, weight, bias, False)

            # Performance test (using more stable method)
            t_mps, std_mps = bench_robust_stable(
                run_mps, warmup=1000, iters=500, runs=210, desc=config_str
            )
            t_torch, _ = bench_robust_stable(
                run_torch, warmup=1000, iters=500, runs=210, desc=config_str
            )

            # Correctness verification
            result_mps = run_mps()
            result_torch = run_torch()
            max_diff = torch.max(torch.abs(result_mps - result_torch)).item()
            is_correct = max_diff < get_tolerance_for(DTYPE)

            speedup = t_torch / t_mps
            std_percent_mps = (std_mps / t_mps) * 100 if t_mps > 0 else 0
            print(
                f"{config_str:<20} {t_mps * 1000:<10.2f} {t_torch * 1000:<12.2f} {speedup:<10.2f} {std_percent_mps:<15.2f} {'✅' if is_correct else '❌':<8}"
            )

            if not is_correct:
                print(f"  ⚠️ Max difference: {max_diff:.6f}")

            # Take a break between different configurations to mitigate temperature effects
            time.sleep(1)

        except Exception as e:
            print(
                f"{config_str:<20} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<15} {'❌':<8}"
            )
            print(f"  Error: {e}")

    # SiLU activation function performance test
    print("\n🔥 SiLU activation function performance test")
    print(
        f"{'Config':<20} {'MPS+SiLU(ms)':<14} {'PyTorch+SiLU(ms)':<17} {'Speedup':<10} {'MPS_StdDev(%)':<15}"
    )
    print("-" * 90)

    # Test activation function with medium scale
    batch, dim, seqlen, width = 2, 128, 256, 4
    config_str = f"{batch}×{dim}×{seqlen}×{width}"

    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
    weight = torch.randn(dim, width, device=device, dtype=DTYPE)
    bias = torch.randn(dim, device=device, dtype=DTYPE)

    try:

        def run_mps_silu():
            return ccmps.causal_conv1d_fwd(
                x.contiguous(), weight.contiguous(), bias.contiguous(), True
            )

        def run_torch_silu():
            return causal_conv1d_reference(x, weight, bias, True)

        t_mps_silu, std_mps_silu = bench_robust_stable(
            run_mps_silu, warmup=1000, iters=400, runs=210, desc=config_str
        )
        t_torch_silu, _ = bench_robust_stable(
            run_torch_silu, warmup=1000, iters=400, runs=210, desc=config_str
        )
        speedup_silu = t_torch_silu / t_mps_silu
        std_percent_mps_silu = (
            (std_mps_silu / t_mps_silu) * 100 if t_mps_silu > 0 else 0
        )

        print(
            f"{config_str:<20} {t_mps_silu * 1000:<14.2f} {t_torch_silu * 1000:<17.2f} {speedup_silu:<10.2f} {std_percent_mps_silu:<15.2f}"
        )

    except Exception as e:
        print(f"{config_str:<20} {'ERROR':<12} {'ERROR':<15} {'ERROR':<10}")
        print(f"  Error: {e}")

    print("\n📊 Performance test completed!")
    print(
        "💡 Tip: Speedup > 1.0 means MPS is faster. StdDev(%) smaller means more stable test results."
    )

if __name__ == "__main__":
    main()
