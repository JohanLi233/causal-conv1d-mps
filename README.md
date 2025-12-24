## causal-conv1d-mps

High-performance CausalConv1D for PyTorch on Apple Silicon using Metal/MPS. It provides optimized forward/backward kernels for causal convolution.

- Backend: Apple Metal Performance Shaders (MPS)
- Languages: Python + C++/Objective-C++ + Metal Shaders
- Precisions: fp32 / fp16 / bf16
- Kernel constraint: current convolution width is fixed to 4 (width=4)


### Features
- High-performance CausalConv1D: MPS backend with native Metal compute.
- Autograd support: provided autograd functions with an optimized two-pass backward for the basic convolution (O(W) complexity).


### Requirements
- macOS (Apple Silicon, macOS 13+ recommended)
- Python ≥ 3.10
- PyTorch ≥ 2.1 (with MPS support)
- Clang/Xcode toolchain (to build the extension)
- Ninja ≥ 1.11 (faster builds)

If `torch.backends.mps.is_available()` returns False, please enable MPS per PyTorch documentation.


### Installation
Run at the repository root:

```bash
uv pip install -e .
```

Notes:
- The extension loads `causal_conv1d.metal` at runtime. It first checks the `CAUSAL_CONV1D_METAL_PATH` environment variable; if not set, it tries to find the file at the repo root or the current working directory.
- If using the package from a non-source directory, set:

```bash
export CAUSAL_CONV1D_METAL_PATH=/absolute/path/to/causal-conv1d-mps/causal_conv1d.metal
```


### Quick Start

#### Basic causal 1D convolution (with autograd)
```python
import torch
import causal_conv1d_mps as ccmps

B, D, T, W = 2, 128, 256, 4
x = torch.randn(B, D, T, device='mps', dtype=torch.bfloat16)
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)

# activation: None / 'silu' / 'swish'
y = ccmps.causal_conv1d(x, weight, bias, activation='silu')
loss = y.mean()
loss.backward()
```

Shapes and device constraints:
- `x`: (B, D, T) on MPS device
- `weight`: (D, W=4), same dtype as `x`, on MPS
- `bias` (optional): (D), same dtype as `x`, on MPS
- Returns: `(B, D, T)`

### Python API Overview
- Autograd-enabled (recommended):
  - `causal_conv1d(x, weight, bias=None, activation=None) -> (B, D, T)`
    - `activation ∈ {None, 'silu', 'swish'}`
- Forward-only (low-level wrappers):
  - `causal_conv1d_fwd(x, weight, bias=None, silu_activation=False) -> (B, D, T)`

Common constraints:
- All tensors must be on the MPS device and share the same dtype
- Only width `W=4` is supported currently
- Tensors must be contiguous
- Some advanced parameters (e.g., `seq_idx`, `initial_states`, `return_final_states`) are not implemented on MPS


### Benchmarks
Run the basic and common scenarios:

```bash
python bench.py --dtype bf16
```

The script does warmup and multiple measured runs; it may take a while. Outputs include MPS/reference times, speedup, and stability statistics per configuration.

Example results:

```text
python bench.py
🚀 Causal Conv1D MPS performance test
Config               MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
--------------------------------------------------------------------------------
1×64×128×4           0.00       0.03         8.83       17.64           ✅
2×128×256×4          0.00       0.04         8.18       18.71           ✅
4×256×512×4          0.01       0.03         3.42       21.24           ✅
1×512×1024×4         0.01       0.04         3.33       13.73           ✅
8×64×128×4           0.00       0.03         7.06       72.24           ✅

🔥 SiLU activation function performance test
Config               MPS+SiLU(ms)   PyTorch+SiLU(ms)  Speedup    MPS_StdDev(%)
------------------------------------------------------------------------------------------
2×128×256×4          0.00           0.05              9.58       53.09

📊 Performance test completed!
💡 Tip: Speedup > 1.0 means MPS is faster. StdDev(%) smaller means more stable test results.
```


### Tests
```bash
python test.py
```


### License
MIT
